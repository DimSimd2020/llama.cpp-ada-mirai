// Mirai S HIP kernels. Rotation, trellis decoding and layouts follow the CPU reference
// and alesha-pro's CUDA port. No NVIDIA PTX or CUDA tensor-core instructions are used.
#include "../ggml-cuda/mirai-s.cuh"
#include <cstring>

namespace {
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
constexpr int GEMV_WARPS = 8;
struct codebook_t {
    float c[5];
};

template <int V, int T, int STEPS, int WORDS, typename ENTRY> struct trellis_format {
    static constexpr int v = V, t = T, steps = STEPS, words = WORDS;
    static constexpr int columns = STEPS * V;  // per packet
    typedef ENTRY entry;
};
typedef trellis_format<4, 8, 16, 1, u8>  fmt_v4t8;
typedef trellis_format<2, 4, 32, 1, u16> fmt_v2t4;
typedef trellis_format<2, 6, 64, 3, u16> fmt_v2t6;

// one 32-row group of a trellis tensor: packets [P][WORDS][32][16 B], then entry states [P][32]
template <typename F> struct trellis_group {
    const uint4 * packets;
    const typename F::entry * entries;
};

template <typename F>
__device__ __forceinline__ trellis_group<F> group_at(const u8 * w, size_t group, int packets_per_row) {
    const size_t packet_bytes = static_cast<size_t>(packets_per_row) * F::words * 512;
    const size_t group_bytes = packet_bytes + static_cast<size_t>(packets_per_row) * 32 * sizeof(typename F::entry);
    const u8 * base = w + group * group_bytes;
    return {reinterpret_cast<const uint4 *>(base), reinterpret_cast<const typename F::entry *>(base + packet_bytes)};
}

__device__ __forceinline__ u32 fmix_hash(u32 state) {
    u32 x = state * 0xCFCCB83Fu + 0x584B4AA3u;
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    return x ^ (x >> 16);
}

// level(b) + 54 for each byte of h, in [0, 111]
__device__ __forceinline__ u32 levels_plus_54(u32 h) {
    const u32 nibble_pairs = (h & 0x33333333u) + ((h >> 2) & 0x33333333u);
    const u32 pairs = (nibble_pairs + (nibble_pairs >> 4)) & 0x0F0F0F0Fu;
    return (pairs << 3) + (((h & 0x0F0F0F0Fu) * 3u) & 0x0F0F0F0Fu);
}


template <int T, int WORDS>
__device__ __forceinline__ u32 symbol_at(const u32 (&bits)[4 * WORDS], int index) {
    const int bit = index * T, word = bit / 32, shift = bit % 32;
    uint64_t joined = bits[word];
    if (shift + T > 32) joined |= static_cast<uint64_t>(bits[word + 1]) << 32;
    return static_cast<u32>(joined >> shift) & ((1u << T) - 1);
}
__device__ __forceinline__ int dp4a_us(u32 a, u32 b, int acc) {
    // Every decoded level is <= 111, so signed dot4 preserves the unsigned levels.
    return ggml_cuda_dp4a(a, b, acc);
}
template <int WORDS>
__device__ __forceinline__ void load_packet(const uint4 * packets, int packet, int lane, u32 (&bits)[4 * WORDS]) {
#pragma unroll
    for (int word = 0; word < WORDS; ++word) {
        const uint4 chunk = __ldg(packets + (static_cast<size_t>(packet) * WORDS + word) * 32 + lane);
        bits[4 * word] = chunk.x;
        bits[4 * word + 1] = chunk.y;
        bits[4 * word + 2] = chunk.z;
        bits[4 * word + 3] = chunk.w;
    }
}

// Replays one packet of a lane's row, handing each 4-column word of levels (level + 54 per byte, columns 4g .. 4g+3 of
// the packet) to use(word, g). Loops are fully unrolled, so g is a compile-time constant inside `use`.
template <typename F, typename Use>
__device__ __forceinline__ void decode_packet(const u32 (&bits)[4 * F::words], u32 state, Use && use) {
    if constexpr (F::v == 4) {
#pragma unroll
        for (int step = 0; step < F::steps; ++step) {
            state = ((state << F::t) | symbol_at<F::t, F::words>(bits, step)) & 0xFFFFu;
            use(levels_plus_54(fmix_hash(state)), step);
        }
    } else {  // two V=2 steps make one word: bytes 0-1 of each step's hash
#pragma unroll
        for (int step = 0; step < F::steps; step += 2) {
            state = ((state << F::t) | symbol_at<F::t, F::words>(bits, step)) & 0xFFFFu;
            const u32 first = fmix_hash(state);
            state = ((state << F::t) | symbol_at<F::t, F::words>(bits, step + 1)) & 0xFFFFu;
            use(levels_plus_54((first & 0xFFFFu) | (fmix_hash(state) << 16)), step / 2);
        }
    }
}

// y = rowscale * (c * s * ((coarse - 54 sum_q0) + (fine - 54 sum_q1) / 254) + sum_r d_r S_r)
// token = {s, sum q0, sum q1, S_0 .. S_3, 0}
__device__ __forceinline__ float output_value(int coarse, int fine, const float * __restrict__ token,
                                              const codebook_t & codebook, float rowscale) {
    coarse -= 54 * __float2int_rn(token[1]);
    fine -= 54 * __float2int_rn(token[2]);
    const float offsets =
        codebook.c[1] * token[3] + codebook.c[2] * token[4] + codebook.c[3] * token[5] + codebook.c[4] * token[6];
    const float dot = codebook.c[0] * token[0] * (static_cast<float>(coarse) + static_cast<float>(fine) * (1.0f / 254.0f));
    return rowscale * (dot + offsets);
}

__device__ __forceinline__ void write_stats(float * out, float step, const int (&coarse)[4], const int (&fine)[4]) {
    out[0] = step;
    out[1] = static_cast<float>(coarse[0] + coarse[1] + coarse[2] + coarse[3]);
    out[2] = static_cast<float>(fine[0] + fine[1] + fine[2] + fine[3]);
#pragma unroll
    for (int r = 0; r < 4; ++r) out[3 + r] = step * (static_cast<float>(coarse[r]) + static_cast<float>(fine[r]) / 254.0f);
    out[7] = 0.0f;
}

__device__ __forceinline__ float block_max(float value, float * shared) {
    for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor(value, offset, 32));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = lane < warps ? shared[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor(value, offset, 32));
    return value;
}

template <int WIDTH, int ORDER>
__global__ void rotate_columns(const float * __restrict__ x, const float * __restrict__ signs,
                               const float * __restrict__ small_q, float * __restrict__ rotated,
                               float * __restrict__ column_max) {
    constexpr int N = WIDTH * ORDER;
    __shared__ float column[WIDTH];
    __shared__ float mix[ORDER];
    __shared__ float shared[32];
    const int out = blockIdx.x;
    const size_t base = static_cast<size_t>(blockIdx.y) * N;
    if (threadIdx.x < ORDER) mix[threadIdx.x] = small_q[out * ORDER + threadIdx.x];
    __syncthreads();
    for (int w = threadIdx.x; w < WIDTH; w += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int c = 0; c < ORDER; ++c) value += x[base + w * ORDER + c] * signs[w * ORDER + c] * mix[c];
        column[w] = value;
    }
    __syncthreads();
#pragma unroll
    for (int stride = 1; stride < WIDTH; stride <<= 1) {
        for (int pair = threadIdx.x; pair < WIDTH / 2; pair += blockDim.x) {
            const int low = (pair / stride) * 2 * stride + pair % stride, high = low + stride;
            const float a = column[low], b = column[high];
            column[low] = a + b;
            column[high] = a - b;
        }
        __syncthreads();
    }
    const float normalization = rsqrtf(static_cast<float>(WIDTH));
    float maximum = 0.0f;
    for (int w = threadIdx.x; w < WIDTH; w += blockDim.x) {
        const float value = column[w] * normalization;
        rotated[base + w * ORDER + out] = value;
        maximum = fmaxf(maximum, fabsf(value));
    }
    maximum = block_max(maximum, shared);
    if (threadIdx.x == 0) column_max[blockIdx.y * ORDER + out] = maximum;
}

template <int WIDTH, int ORDER>
__global__ void __launch_bounds__(1024) quantize_rows(const float * __restrict__ rotated, const float * __restrict__ column_max,
                                                      uint2 * __restrict__ q, float * __restrict__ stats) {
    constexpr int N = WIDTH * ORDER;
    __shared__ int sums[32][8];
    const size_t base = static_cast<size_t>(blockIdx.x) * N;
    float maximum = 0.0f;
#pragma unroll
    for (int c = 0; c < ORDER; ++c) maximum = fmaxf(maximum, column_max[blockIdx.x * ORDER + c]);
    const float step = maximum > 0.0f ? maximum / 127.0f : 1.0f, inverse = 1.0f / step;
    int local[8] = {};  // per residue class: sum q0 (0..3), sum q1 (4..7)
    for (int group = threadIdx.x; group < N / 4; group += blockDim.x) {
        const float4 values = reinterpret_cast<const float4 *>(rotated + base)[group];
        const float v[4] = {values.x, values.y, values.z, values.w};
        u32 plane0 = 0, plane1 = 0;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float scaled = v[r] * inverse;
            const int coarse = min(127, max(-127, __float2int_rn(scaled)));
            const int fine = min(127, max(-127, __float2int_rn((scaled - coarse) * 254.0f)));
            plane0 |= (static_cast<u32>(coarse) & 0xFFu) << (8 * r);
            plane1 |= (static_cast<u32>(fine) & 0xFFu) << (8 * r);
            local[r] += coarse;
            local[4 + r] += fine;
        }
        q[static_cast<size_t>(blockIdx.x) * (N / 4) + group] = make_uint2(plane0, plane1);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
        for (int offset = 16; offset > 0; offset >>= 1) local[i] += __shfl_xor(local[i], offset, 32);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < 8; ++i) sums[warp][i] = local[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        int coarse[4] = {}, fine[4] = {};
        for (int w = 0; w < warps; ++w)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                coarse[r] += sums[w][r];
                fine[r] += sums[w][4 + r];
            }
        write_stats(stats + static_cast<size_t>(blockIdx.x) * 8, step, coarse, fine);
    }
}

__device__ __forceinline__ float hadamard32(float x, int lane) {
#pragma unroll
    for (int half = 1; half < 32; half <<= 1) {
        const float other = __shfl_xor(x, half, 32);
        x = (lane & half) ? other - x : x + other;
    }
    return x / sqrtf(32.0f);
}

__global__ void __launch_bounds__(256) head_input(const float * __restrict__ x, const float * __restrict__ signs,
                                                  half * __restrict__ x_rot, int K) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t base = static_cast<size_t>(blockIdx.x) * K;
    for (int block = warp; block < K / 32; block += 8) {
        const int column = 32 * block + lane;
        x_rot[base + column] = __float2half_rn(hadamard32(x[base + column] * signs[column], lane));
    }
}

template <typename F, int TILE>
__global__ void trellis_mul(const u8 * w, int packets, const float * scale, const uint2 * q,
                           const float * stats, codebook_t codebook, float * y, int K, int N, int T) {
    __shared__ int partial[TILE][2][GEMV_WARPS][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const trellis_group<F> tape = group_at<F>(w, blockIdx.x, packets);
    int coarse[TILE] = {}, fine[TILE] = {};
    for (int p = warp; p < packets; p += GEMV_WARPS) {
        u32 bits[4 * F::words];
        load_packet<F::words>(tape.packets, p, lane, bits);
        decode_packet<F>(bits, __ldg(tape.entries + p * 32 + lane), [&](u32 levels, int g) {
            #pragma unroll
            for (int t = 0; t < TILE; ++t) {
                const int token = blockIdx.y * TILE + t;
                if (token < T) {
                    const uint2 value = q[static_cast<size_t>(token) * (K / 4) + p * F::columns / 4 + g];
                    coarse[t] = dp4a_us(levels, value.x, coarse[t]);
                    fine[t] = dp4a_us(levels, value.y, fine[t]);
                }
            }
        });
    }
    #pragma unroll
    for (int t = 0; t < TILE; ++t) {
        partial[t][0][warp][lane] = coarse[t];
        partial[t][1][warp][lane] = fine[t];
    }
    __syncthreads();
    if (warp != 0) return;
    const int row = blockIdx.x * 32 + lane;
    #pragma unroll
    for (int t = 0; t < TILE; ++t) {
        const int token = blockIdx.y * TILE + t;
        if (token >= T) continue;
        int c = 0, f = 0;
        #pragma unroll
        for (int s = 0; s < GEMV_WARPS; ++s) {
            c += partial[t][0][s][lane];
            f += partial[t][1][s][lane];
        }
        y[static_cast<size_t>(token) * N + row] = output_value(c, f, stats + token * 8, codebook, scale[row]);
    }
}

__global__ void head_mul(const u8 * w, const half * x, const float * scale, const float * ladder,
                         float * y, int K, int N) {
    __shared__ float partial[GEMV_WARPS][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t pairs = K / 128;
    const size_t group_bytes = pairs * 32 * (3 * 16 + 1);
    const u8 * base = w + static_cast<size_t>(blockIdx.x) * group_bytes;
    const half * xt = x + static_cast<size_t>(blockIdx.y) * K;
    float sum = 0.0f;
    for (size_t pair = warp; pair < pairs; pair += GEMV_WARPS) {
        u32 bits[12];
        #pragma unroll
        for (int part = 0; part < 3; ++part) {
            const uint4 v = reinterpret_cast<const uint4 *>(base)[(pair * 3 + part) * 32 + lane];
            bits[part * 4] = v.x; bits[part * 4 + 1] = v.y;
            bits[part * 4 + 2] = v.z; bits[part * 4 + 3] = v.w;
        }
        const u8 steps = base[pairs * 3 * 32 * 16 + pair * 32 + lane];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float step = __half2float(__float2half_rn(ladder[h ? steps >> 4 : steps & 15]));
            #pragma unroll
            for (int j = 0; j < 64; ++j) {
                const int code = symbol_at<3, 3>(bits, h * 64 + j);
                const float weight = __half2float(__float2half_rn((2 * code - 7) * step));
                sum += weight * __half2float(xt[pair * 128 + h * 64 + j]);
            }
        }
    }
    partial[warp][lane] = sum;
    __syncthreads();
    if (warp != 0) return;
    sum = 0.0f;
    #pragma unroll
    for (int s = 0; s < GEMV_WARPS; ++s) sum += partial[s][lane];
    const int row = blockIdx.x * 32 + lane;
    y[static_cast<size_t>(blockIdx.y) * N + row] = scale[row] * sum;
}

template <int WIDTH, int ORDER>
void quantize(ggml_backend_cuda_context & ctx, const float * x, const float * rot, uint2 * q, float * stats, int T) {
    ggml_cuda_pool_alloc<float> rotated(ctx.pool(), static_cast<size_t>(T) * WIDTH * ORDER);
    ggml_cuda_pool_alloc<float> maxima(ctx.pool(), T * ORDER);
    rotate_columns<WIDTH, ORDER><<<dim3(ORDER, T), 256, 0, ctx.stream()>>>(x, rot, rot + WIDTH * ORDER, rotated.get(), maxima.get());
    quantize_rows<WIDTH, ORDER><<<T, 256, 0, ctx.stream()>>>(rotated.get(), maxima.get(), q, stats);
}

template <typename F>
void mul(ggml_backend_cuda_context & ctx, const u8 * w, const float * scale, const uint2 * q,
         const float * stats, codebook_t cb, float * y, int K, int N, int T) {
    const int P = K / F::columns;
    if (T == 1) {
        trellis_mul<F, 1><<<dim3(N / 32, T), 32 * GEMV_WARPS, 0, ctx.stream()>>>(w, P, scale, q, stats, cb, y, K, N, T);
    } else {
        trellis_mul<F, 4><<<dim3(N / 32, (T + 3) / 4), 32 * GEMV_WARPS, 0, ctx.stream()>>>(w, P, scale, q, stats, cb, y, K, N, T);
    }
}
} // namespace

bool ggml_cuda_mirai_supports_op(const ggml_tensor * op) {
    if (op->op == GGML_OP_MIRAI_QUANTIZE) {
        const int64_t K = op->src[0]->ne[0];
        if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) return false;
        if (ggml_get_op_params_i32(op, 0)) return K % 32 == 0;
        const int order = ggml_get_op_params_i32(op, 1);
        return (K == 5120 && order == 5) || (K == 6144 && order == 3) || (K == 17408 && order == 17);
    }
    const ggml_tensor * w = op->src[0];
    if (w->ne[1] % 32 != 0) return false;
    switch (w->type) {
        case GGML_TYPE_MS_V4T8:
        case GGML_TYPE_MS_V2T4: return w->ne[0] % 64 == 0;
        case GGML_TYPE_MS_V2T6:
        case GGML_TYPE_MS_I3: return w->ne[0] % 128 == 0;
        default: return false;
    }
}

void ggml_cuda_op_mirai_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int K = dst->src[0]->ne[0], T = ggml_nrows(dst->src[0]);
    const float * x = static_cast<const float *>(dst->src[0]->data);
    const float * rot = static_cast<const float *>(dst->src[1]->data);
    if (ggml_get_op_params_i32(dst, 0)) {
        head_input<<<T, 256, 0, ctx.stream()>>>(x, rot, static_cast<half *>(dst->data), K);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    uint2 * q = static_cast<uint2 *>(dst->data);
    float * stats = reinterpret_cast<float *>(q + static_cast<size_t>(T) * (K / 4));
    switch (K) {
        case 5120: quantize<1024, 5>(ctx, x, rot, q, stats, T); break;
        case 6144: quantize<2048, 3>(ctx, x, rot, q, stats, T); break;
        case 17408: quantize<1024, 17>(ctx, x, rot, q, stats, T); break;
        default: GGML_ABORT("Mirai S HIP: unsupported rotation");
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_mirai_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0], * x = dst->src[1];
    const int K = w->ne[0], N = w->ne[1], T = x->ne[1];
    const u8 * wd = static_cast<const u8 *>(w->data);
    const float * scale = static_cast<const float *>(dst->src[2]->data);
    float * y = static_cast<float *>(dst->data);
    if (w->type == GGML_TYPE_MS_I3) {
        head_mul<<<dim3(N / 32, T), 32 * GEMV_WARPS, 0, ctx.stream()>>>(wd, static_cast<const half *>(x->data), scale,
            static_cast<const float *>(dst->src[3]->data), y, K, N);
    } else {
        const uint2 * q = static_cast<const uint2 *>(x->data);
        const float * stats = reinterpret_cast<const float *>(q + static_cast<size_t>(T) * (K / 4));
        codebook_t cb;
        memcpy(cb.c, dst->op_params, sizeof(cb.c));
        switch (w->type) {
            case GGML_TYPE_MS_V4T8: mul<fmt_v4t8>(ctx, wd, scale, q, stats, cb, y, K, N, T); break;
            case GGML_TYPE_MS_V2T4: mul<fmt_v2t4>(ctx, wd, scale, q, stats, cb, y, K, N, T); break;
            case GGML_TYPE_MS_V2T6: mul<fmt_v2t6>(ctx, wd, scale, q, stats, cb, y, K, N, T); break;
            default: GGML_ABORT("Mirai S HIP: unsupported weight type");
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

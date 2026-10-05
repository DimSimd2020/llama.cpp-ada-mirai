// Mirai S linears on CUDA, ported from the kernels of Mirai's vLLM plugin (mirai_s 0.2.1, Apache-2.0) to ggml: f32
// activations in and out, weights in the GGUF layout of tools/mirai-s/convert_mirai_s_to_gguf.py, the row scale fused
// into the output. The math and layouts are documented in ggml-cpu/mirai-s.cpp, the reference implementation.
//
//   ggml_mirai_quantize   x_rot = R (signs * x) as two int8 planes per token plus per-token sums
//                         (<= 16 tokens: rotate_columns + quantize_rows; longer: transform, one CTA per token)
//   ggml_mirai_mul_mat    1 token: dp4a GEMV; <= 384 tokens: int8 tensor-core MMA on decoded weights;
//                         longer: weights decoded to int8 levels, cuBLAS int8 GEMM, then the output epilogue
//                         MS_I3 head: fp16 tensor-core MMA against x_rot = H32(signs * x)
// All trellis paths compute the same two exact int32 dot products per (token, row), so a token's output does not
// depend on the batch it runs in.

#include "mirai-s.cuh"

#include <cublasLt.h>

#include <cstring>

#include <map>
#include <mutex>
#include <tuple>

namespace {

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef signed char s8;

constexpr int GEMV_WARPS = 16;  // column slices per 32-row CTA
constexpr int MMA_WARPS = 8;
constexpr int SLAB = 20;        // u32 per slab row: 16 words of levels + 4 of padding
// path thresholds (tokens); GGML_MIRAI_MMA_TOKENS / GGML_MIRAI_SPLIT_TOKENS override them for experiments
static int env_tokens(const char * name, int fallback) {
    const char * value = getenv(name);
    return value ? atoi(value) : fallback;
}
static const int MMA_TOKENS   = env_tokens("GGML_MIRAI_MMA_TOKENS", 384);  // beyond this, levels + cuBLAS int8 GEMM
static const int SPLIT_TOKENS = env_tokens("GGML_MIRAI_SPLIT_TOKENS", 16);  // up to this, rotate_columns + quantize_rows
// long inputs decode the weights to int8 levels in chunks of rows; this bounds the chunk's buffers (int8 levels plus
// int32 products) in MiB. On a 4070: 64 -> 1038, 128 -> 1089, 256 -> 1086 tok/s on a 16.8k prompt; 128 is the knee.
static const int LEVELS_MIB   = env_tokens("GGML_MIRAI_LEVELS_MIB", 128);
constexpr int HEAD_WARPS = 8;
constexpr int HEAD_SLAB = 72;   // halves per head slab row: 64 columns + 8 padding

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

// acc + sum_i a.u8[i] * b.s8[i]
__device__ __forceinline__ int dp4a_us(u32 a, u32 b, int acc) {
    int result;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(acc));
    return result;
}

template <int T, int WORDS>
__device__ __forceinline__ u32 symbol_at(const u32 (&bits)[4 * WORDS], int index) {
    const int bit = index * T;
    const int word = bit / 32, shift = bit % 32;
    if (shift + T <= 32) return (bits[word] >> shift) & ((1u << T) - 1);
    return __funnelshift_r(bits[word], bits[word + 1], shift) & ((1u << T) - 1);
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
            use(levels_plus_54(__byte_perm(first, fmix_hash(state), 0x5410)), step / 2);
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

// ---------------------------------------------------------------------------------------------------------------
// Rotation and quantization, per input shape (WIDTH x ORDER = 1024 x 5, 2048 x 3, 1024 x 17):
//   s = max |x_rot| / 127, q0 = round(x / s), q1 = round((x / s - q0) * 254).
// Up to 16 tokens: rotate_columns grid (ORDER, tokens) does column o's small_q mix, its WIDTH-point Walsh-Hadamard and
// the column max; quantize_rows grid (tokens) packs q[m, g] as a uint2 (plane 0, plane 1) of columns 4g .. 4g+3.
// Longer inputs: transform, one CTA of 512 threads per token. stats[m] = {s, sum q0, sum q1, residue 0..3, 0}.

__device__ __forceinline__ void write_stats(float * out, float step, const int (&coarse)[4], const int (&fine)[4]) {
    out[0] = step;
    out[1] = static_cast<float>(coarse[0] + coarse[1] + coarse[2] + coarse[3]);
    out[2] = static_cast<float>(fine[0] + fine[1] + fine[2] + fine[3]);
#pragma unroll
    for (int r = 0; r < 4; ++r) out[3 + r] = step * (static_cast<float>(coarse[r]) + static_cast<float>(fine[r]) / 254.0f);
    out[7] = 0.0f;
}

__device__ __forceinline__ float block_max(float value, float * shared) {
    for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = lane < warps ? shared[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
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
        for (int offset = 16; offset > 0; offset >>= 1) local[i] += __shfl_xor_sync(0xffffffffu, local[i], offset);
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

// Butterfly at lane distance `mask`: the lower index of each pair keeps a + b, the upper a - b.
__device__ __forceinline__ float butterfly(float value, int lane, int mask) {
    const float other = __shfl_xor_sync(0xffffffffu, value, mask);
    return (lane & mask) ? other - value : value + other;
}

// transform: grid (tokens), 512 threads. Every input is read once and kept in registers; for each small_q column the
// thread mixes its points, the Walsh-Hadamard runs as warp shuffles around one shared-memory transpose, and the values
// stay in registers for the max and the quantization (same summation and butterfly order as rotate_columns).
template <int WIDTH, int ORDER>
__global__ void __launch_bounds__(512) transform(const float * __restrict__ x, const float * __restrict__ signs,
                                                 const float * __restrict__ small_q, uint2 * __restrict__ q,
                                                 float * __restrict__ stats) {
    constexpr int PER = WIDTH / 512, N = WIDTH * ORDER;  // points per thread per column
    __shared__ float values[WIDTH];
    __shared__ alignas(16) s8 staged[2][N];
    __shared__ float mix[ORDER * ORDER];
    __shared__ int partial[16][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t token = blockIdx.x;
    for (int i = threadIdx.x; i < ORDER * ORDER; i += 512) mix[i] = small_q[i];

    // phase 1: the thread's points are h = 32 * (warp + 16 * i) + lane
    float inputs[PER][ORDER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int h = 32 * (warp + 16 * i) + lane;
#pragma unroll
        for (int c = 0; c < ORDER; ++c) inputs[i][c] = x[token * N + h * ORDER + c] * signs[h * ORDER + c];
    }
    __syncthreads();

    // phase 2: WIDTH 1024 holds h = 32 * lane + warp + 16 * i; WIDTH 2048 holds h = 32 * (lane + 32 * (i % 2)) +
    // warp + 16 * (i / 2)
    const auto phase2_point = [&](int i) {
        return WIDTH == 1024 ? 32 * lane + warp + 16 * i : 32 * (lane + 32 * (i % 2)) + warp + 16 * (i / 2);
    };
    const float normalization = rsqrtf(static_cast<float>(WIDTH));
    float out[ORDER][PER];
    float maximum = 0.0f;
#pragma unroll
    for (int o = 0; o < ORDER; ++o) {
        float element[PER];
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            float value = 0.0f;
#pragma unroll
            for (int c = 0; c < ORDER; ++c) value += inputs[i][c] * mix[o * ORDER + c];
            element[i] = value;
        }
#pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1)  // strides 1..16: lanes hold consecutive h
#pragma unroll
            for (int i = 0; i < PER; ++i) element[i] = butterfly(element[i], lane, mask);
        __syncthreads();  // the previous column is done reading values
#pragma unroll
        for (int i = 0; i < PER; ++i) values[32 * (warp + 16 * i) + lane] = element[i];
        __syncthreads();
#pragma unroll
        for (int i = 0; i < PER; ++i) element[i] = values[phase2_point(i)];
#pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1)  // strides 32..512: the lane bits of h
#pragma unroll
            for (int i = 0; i < PER; ++i) element[i] = butterfly(element[i], lane, mask);
        if constexpr (WIDTH == 2048) {  // stride 1024: the register pairs (i, i + 1)
#pragma unroll
            for (int i = 0; i < PER; i += 2) {
                const float low = element[i], high = element[i + 1];
                element[i] = low + high;
                element[i + 1] = low - high;
            }
        }
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            out[o][i] = element[i] * normalization;
            maximum = fmaxf(maximum, fabsf(out[o][i]));
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    __syncthreads();  // values is reused for the max
    if (lane == 0) values[warp] = maximum;
    __syncthreads();
    maximum = 0.0f;
#pragma unroll
    for (int w = 0; w < 16; ++w) maximum = fmaxf(maximum, values[w]);
    const float step = maximum > 0.0f ? maximum / 127.0f : 1.0f, inverse = 1.0f / step;

    int local[8] = {};  // per residue class: sum q0 (0..3), sum q1 (4..7)
#pragma unroll
    for (int o = 0; o < ORDER; ++o)
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int column = phase2_point(i) * ORDER + o;
            const float scaled = out[o][i] * inverse;
            const int coarse = min(127, max(-127, __float2int_rn(scaled)));
            const int fine = min(127, max(-127, __float2int_rn((scaled - coarse) * 254.0f)));
            staged[0][column] = static_cast<s8>(coarse);
            staged[1][column] = static_cast<s8>(fine);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                local[r] += column % 4 == r ? coarse : 0;
                local[4 + r] += column % 4 == r ? fine : 0;
            }
        }
#pragma unroll
    for (int k = 0; k < 8; ++k)
        for (int offset = 16; offset > 0; offset >>= 1) local[k] += __shfl_xor_sync(0xffffffffu, local[k], offset);
    if (lane == 0)
#pragma unroll
        for (int k = 0; k < 8; ++k) partial[warp][k] = local[k];
    __syncthreads();
    uint2 * destination = q + token * (N / 4);
    for (int i = threadIdx.x; i < N / 4; i += 512)
        destination[i] = make_uint2(reinterpret_cast<const u32 *>(staged[0])[i], reinterpret_cast<const u32 *>(staged[1])[i]);
    if (threadIdx.x == 0) {
        int coarse[4] = {}, fine[4] = {};
        for (int w = 0; w < 16; ++w)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                coarse[r] += partial[w][r];
                fine[r] += partial[w][4 + r];
            }
        write_stats(stats + token * 8, step, coarse, fine);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// gemv: one token. One CTA per 32-row group, GEMV_WARPS column slices; each state feeds two dp4a.

template <typename F>
__global__ void __launch_bounds__(32 * GEMV_WARPS)
    gemv(const u8 * __restrict__ w, int packets_per_row, const float * __restrict__ rowscale,
         const uint2 * __restrict__ q, const float * __restrict__ stats, codebook_t codebook, float * __restrict__ y) {
    __shared__ int partial[2][GEMV_WARPS][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const trellis_group<F> tape = group_at<F>(w, blockIdx.x, packets_per_row);

    int coarse = 0, fine = 0;
    for (int packet = warp; packet < packets_per_row; packet += GEMV_WARPS) {
        u32 bits[4 * F::words];
        load_packet<F::words>(tape.packets, packet, lane, bits);
        const uint2 * x = q + packet * F::columns / 4;
        decode_packet<F>(bits, __ldg(tape.entries + packet * 32 + lane), [&](u32 levels, int g) {
            const uint2 value = __ldg(x + g);
            coarse = dp4a_us(levels, value.x, coarse);
            fine = dp4a_us(levels, value.y, fine);
        });
    }
    partial[0][warp][lane] = coarse;
    partial[1][warp][lane] = fine;
    __syncthreads();
    if (warp != 0) return;
    coarse = fine = 0;
#pragma unroll
    for (int i = 0; i < GEMV_WARPS; ++i) {
        coarse += partial[0][i][lane];
        fine += partial[1][i][lane];
    }
    const size_t row = static_cast<size_t>(blockIdx.x) * 32 + lane;
    y[row] = output_value(coarse, fine, stats, codebook, rowscale[row]);
}

// ---------------------------------------------------------------------------------------------------------------
// mma: up to NT tokens per CTA (grid.y tiles the batch). Same CTA shape and decode as gemv (lane = row, MMA_WARPS column
// slices), but each warp parks 64 decoded columns of its 32 rows in a shared-memory slab and reads them back as
// m16n8k32 A fragments (ldmatrix). One u8 x s8 MMA takes 16 rows x 32 columns against 8 tokens of one activation plane.
// NT = 64 CTAs take 64 rows, two row groups of 4 warps each walking the same packets; warps 2i and 2i + 1 multiply both
// of their slabs, one against tokens 0-31 and the other against 32-63.

__device__ __forceinline__ void mma_u8s8(int (&c)[4], const u32 (&a)[4], u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A fragment of rows 0-15 x 32 columns (bytes) at `tile`: lane i addresses row i % 16, bytes 16 * (i / 16) of it.
__device__ __forceinline__ void load_a(u32 (&a)[4], const u32 * tile, int lane) {
    const u32 address = static_cast<u32>(__cvta_generic_to_shared(tile + (lane & 15) * SLAB + 4 * (lane >> 4)));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(address));
}

template <typename F, int NT>
__global__ void __launch_bounds__(32 * MMA_WARPS, 2)
    mma(const u8 * __restrict__ w, int packets_per_row, const float * __restrict__ rowscale,
        const uint2 * __restrict__ q, int groups_per_token, int tokens, const float * __restrict__ stats,
        codebook_t codebook, float * __restrict__ y, int y_stride) {
    constexpr bool PAIRED = NT == 64;
    constexpr int GROUPS = PAIRED ? 2 : 1, SLICES = MMA_WARPS / GROUPS;  // row groups per CTA, warps per row group
    constexpr int TILES = (PAIRED ? 32 : NT) / 8, PACKET_WORDS = F::columns / 4;  // a V2 T6 packet is two slabs
    __shared__ alignas(16) u32 slabs[MMA_WARPS][32 * SLAB];
    __shared__ int sums[2][NT][32];  // [plane][token][row], one row group at a time
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, row_group = warp / SLICES;
    const size_t group = static_cast<size_t>(blockIdx.x) * GROUPS + row_group;
    const trellis_group<F> tape = group_at<F>(w, group, packets_per_row);
    const int first_token = blockIdx.y * NT, warp_token = PAIRED ? 32 * (warp & 1) : 0;  // this warp's first, in the CTA
    u32 * slab = slabs[warp];
    for (int i = threadIdx.x; i < 2 * NT * 32; i += blockDim.x) (&sums[0][0][0])[i] = 0;

    // acc[half][tile][plane]: C fragment of rows 16 * half .. + 15 x tokens warp_token + 8 * tile .. + 7
    int acc[2][TILES][2][4] = {};
    const auto multiply = [&](const u32 * levels, int first_group) {
#pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk) {
            u32 a[2][4];
#pragma unroll
            for (int half = 0; half < 2; ++half) load_a(a[half], levels + 16 * half * SLAB + 8 * chunk, lane);
#pragma unroll
            for (int tile = 0; tile < TILES; ++tile) {
                const int token = first_token + warp_token + 8 * tile + (lane >> 2);
                uint2 low = make_uint2(0, 0), high = make_uint2(0, 0);
                if (token < tokens) {
                    const uint2 * x = q + static_cast<size_t>(token) * groups_per_token + first_group + 8 * chunk + (lane & 3);
                    low = __ldg(x);
                    high = __ldg(x + 4);
                }
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    mma_u8s8(acc[half][tile][0], a[half], low.x, high.x);
                    mma_u8s8(acc[half][tile][1], a[half], low.y, high.y);
                }
            }
        }
    };

    for (int packet = warp % SLICES; packet < packets_per_row; packet += SLICES) {
        u32 bits[4 * F::words];
        load_packet<F::words>(tape.packets, packet, lane, bits);
        u32 quad[4];
        decode_packet<F>(bits, __ldg(tape.entries + packet * 32 + lane), [&](u32 word, int g) {
            quad[g % 4] = word;
            if (g % 4 == 3)
                reinterpret_cast<uint4 *>(slab + lane * SLAB)[(g % 16) / 4] = make_uint4(quad[0], quad[1], quad[2], quad[3]);
            if (g % 16 != 15) return;
            // slab full: multiply, then let every lane finish reading before it is overwritten
            if constexpr (PAIRED) {  // the same for both warps of the pair; packet ^ 1 is the partner's
                asm volatile("bar.sync %0, 64;" ::"r"(1 + warp / 2) : "memory");
                multiply(slab, packet * PACKET_WORDS + g - 15);
                multiply(slabs[warp ^ 1], (packet ^ 1) * PACKET_WORDS + g - 15);
                asm volatile("bar.sync %0, 64;" ::"r"(1 + warp / 2) : "memory");
            } else {
                __syncwarp();
                multiply(slab, packet * PACKET_WORDS + g - 15);
                __syncwarp();
            }
        });
    }

    // One row group at a time: C fragment f holds row lane / 4 (+ 8 if f >= 2), token 2 * (lane % 4) + f % 2.
    for (int round = 0; round < GROUPS; ++round) {
        __syncthreads();
        if (row_group == round)
#pragma unroll
            for (int half = 0; half < 2; ++half)
#pragma unroll
                for (int tile = 0; tile < TILES; ++tile)
#pragma unroll
                    for (int plane = 0; plane < 2; ++plane)
#pragma unroll
                        for (int f = 0; f < 4; ++f)
                            atomicAdd(&sums[plane][warp_token + 8 * tile + 2 * (lane & 3) + (f & 1)]
                                          [16 * half + (lane >> 2) + 8 * (f >> 1)],
                                      acc[half][tile][plane][f]);
        __syncthreads();
        for (int i = threadIdx.x; i < NT * 32; i += blockDim.x) {
            const int t = i / 32, r = i % 32, token = first_token + t;
            const size_t row = (static_cast<size_t>(blockIdx.x) * GROUPS + round) * 32 + r;
            if (token < tokens)
                y[static_cast<size_t>(token) * y_stride + row] =
                    output_value(sums[0][t][r], sums[1][t][r], stats + token * 8, codebook, rowscale[row]);
            sums[0][t][r] = sums[1][t][r] = 0;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Long inputs: levels writes W_rot as level + 54 bytes ([rows][columns], all in 0..111, so also valid int8) for a
// cuBLAS int8 GEMM against both activation planes; prefill_output finishes its int32 products like the kernels above.

template <typename F>
__global__ void levels(const u8 * __restrict__ w, int packets_per_row, size_t first_group, u32 * __restrict__ out,
                       int groups_per_row) {
    const int lane = threadIdx.x & 31;
    const trellis_group<F> tape = group_at<F>(w, first_group + blockIdx.x, packets_per_row);
    const size_t row = static_cast<size_t>(blockIdx.x) * 32 + lane;
    // each warp takes a run of consecutive packets, so every lane writes its row front to back
    const int warps = blockDim.x >> 5, per_warp = (packets_per_row + warps - 1) / warps;
    const int first_packet = (threadIdx.x >> 5) * per_warp, end_packet = min(packets_per_row, first_packet + per_warp);
    for (int packet = first_packet; packet < end_packet; ++packet) {
        u32 bits[4 * F::words];
        load_packet<F::words>(tape.packets, packet, lane, bits);
        uint4 * dst = reinterpret_cast<uint4 *>(out + row * groups_per_row + packet * (F::columns / 4));
        u32 quad[4];
        decode_packet<F>(bits, __ldg(tape.entries + packet * 32 + lane), [&](u32 word, int g) {
            quad[g % 4] = word;
            if (g % 4 == 3) dst[g / 4] = make_uint4(quad[0], quad[1], quad[2], quad[3]);
        });
    }
}

// q words [tokens][K / 4] (plane 0, plane 1) -> planes [2 * tokens][K]: plane 0 of every token, then plane 1
__global__ void split_planes(const uint2 * __restrict__ q, int groups_per_token, int tokens, u32 * __restrict__ planes) {
    const int token = blockIdx.y;
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < groups_per_token; g += gridDim.x * blockDim.x) {
        const uint2 value = q[static_cast<size_t>(token) * groups_per_token + g];
        planes[static_cast<size_t>(token) * groups_per_token + g] = value.x;
        planes[(static_cast<size_t>(tokens) + token) * groups_per_token + g] = value.y;
    }
}

// products: [2 * tokens][rows] int32, plane 0 of every token first, then plane 1. Grid (rows / 256, tokens).
// one_plane (GGML_MIRAI_PREFILL_PLANES=1): only plane 0 was multiplied; the fine term is zeroed by handing
// output_value a `fine` equal to its own offset (54 * sum q1), so the activations act as plain per-token int8
// (s = max|x| / 127, round to nearest). The residue sums in stats keep their full precision.
__global__ void prefill_output(const int * __restrict__ products, int tokens, int rows, const float * __restrict__ rowscale,
                               const float * __restrict__ stats, codebook_t codebook, float * __restrict__ y, int y_stride,
                               int one_plane) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, token = blockIdx.y;
    if (row >= rows) return;
    const size_t index = static_cast<size_t>(token) * rows + row;
    const int fine = one_plane ? 54 * __float2int_rn(stats[token * 8 + 2]) : products[static_cast<size_t>(tokens) * rows + index];
    y[static_cast<size_t>(token) * y_stride + row] = output_value(products[index], fine, stats + token * 8, codebook, rowscale[row]);
}

// ---------------------------------------------------------------------------------------------------------------
// The MS_I3 output head: W[v, :] = signs * H32((2c - 7) * row_scale[v] * ladder[group]), groups of 64 columns.
// head_input: x_rot = H32(signs * x) per token, as fp16; then logits[v] = row_scale[v] * sum_j W'[v, j] x_rot[j] with
// W'[v, j] = (2c - 7) * ladder[group], because H32 is symmetric.

__device__ __forceinline__ float hadamard32(float x, int lane) {
#pragma unroll
    for (int half = 1; half < 32; half <<= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, x, half);
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

__device__ __forceinline__ void mma_f16(float (&c)[4], const u32 (&a)[4], u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two I3 weights as fp16, (2c - 7) * step: `bits` holds c0 in bits 0-2 and c1 in bits 3-5. x * 16386 puts 2 c0 at bit
// 1 and 2 c1 at bit 17, so with 0x6400 each half reads 1024 + 2c exactly; minus 1031 is 2c - 7, and the multiply by
// the ladder step is the one rounding.
__device__ __forceinline__ u32 weight_pair(u32 bits, u32 step2) {
    u32 pair = ((bits & 63u) * 16386u & 0x000E000Eu) | 0x64006400u, value;
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(value) : "r"(pair), "r"(0xE407E407u));  // 0xE407 = -1031
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(value) : "r"(value), "r"(step2));
    return value;
}

// Each warp owns 32 vocabulary rows (one 32-row group: codes [pair][part][row][16 B], then ladder bytes [pair][row]).
// Per 64-column group, lane = row decodes (2c - 7) * ladder into a shared-memory slab; the warp reads it back as
// 16 x 16 A fragments (ldmatrix) and runs m16n8k16 MMAs against x_rot with fp32 accumulation.
template <int NT>
__global__ void __launch_bounds__(32 * HEAD_WARPS)
    head_mma(const half * __restrict__ x_rot, int tokens, const u8 * __restrict__ w, int K,
             const float * __restrict__ row_scales, const float * __restrict__ ladder, float * __restrict__ logits,
             int vocab) {
    constexpr int TILES = NT / 8;
    __shared__ alignas(16) u16 slabs[HEAD_WARPS][32 * HEAD_SLAB];
    __shared__ float steps[16];
    if (threadIdx.x < 16) steps[threadIdx.x] = ladder[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t block = static_cast<size_t>(blockIdx.x) * HEAD_WARPS + warp, first_row = block * 32;
    if (first_row >= static_cast<size_t>(vocab)) return;
    const int pairs = K / 128;
    const u8 * base = w + block * 32 * (static_cast<size_t>(pairs) * 49);
    const uint4 * row_codes = reinterpret_cast<const uint4 *>(base) + lane;
    const u8 * row_ladder = base + static_cast<size_t>(pairs) * 3 * 32 * 16 + lane;
    const int first_token = blockIdx.y * NT;
    u16 * slab = slabs[warp];
    u32 carry[6];  // the second group of a loaded pair

    // acc[half][tile]: C fragment of rows 16 * half .. + 15 x tokens 8 * tile .. + 7
    float acc[2][TILES][4] = {};
    uint4 next[3];  // the codes of the next two groups (48 bytes) and their ladder byte, loaded one step ahead
    u8 next_ladder = row_ladder[0], ladder_byte = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) next[i] = __ldg(row_codes + 32 * i);
    for (int group = 0; group < K / 64; ++group) {
        u32 words[6];
        if (group % 2 == 0) {
            const uint4 chunk[3] = {next[0], next[1], next[2]};
            ladder_byte = next_ladder;
            if (group + 2 < K / 64) {
#pragma unroll
                for (int i = 0; i < 3; ++i) next[i] = __ldg(row_codes + 32 * (3 * (group / 2 + 1) + i));
                next_ladder = row_ladder[32 * (group / 2 + 1)];
            }
            const u32 * flat = reinterpret_cast<const u32 *>(chunk);
#pragma unroll
            for (int i = 0; i < 6; ++i) words[i] = flat[i];
            carry[0] = chunk[1].z, carry[1] = chunk[1].w, carry[2] = chunk[2].x;
            carry[3] = chunk[2].y, carry[4] = chunk[2].z, carry[5] = chunk[2].w;
        } else {
#pragma unroll
            for (int i = 0; i < 6; ++i) words[i] = carry[i];
        }
        u16 step_half;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(step_half) : "f"(steps[group % 2 ? ladder_byte >> 4 : ladder_byte & 15]));
        const u32 step2 = static_cast<u32>(step_half) * 0x10001u;
        u32 packed[32];
#pragma unroll
        for (int j = 0; j < 64; j += 2) {
            const int bit = 3 * j;
            packed[j / 2] = weight_pair(__funnelshift_r(words[bit / 32], words[min(bit / 32 + 1, 5)], bit % 32), step2);
        }
        __syncwarp();  // the previous group's ldmatrix reads are done
        uint4 * out = reinterpret_cast<uint4 *>(slab + lane * HEAD_SLAB);
#pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = make_uint4(packed[4 * i], packed[4 * i + 1], packed[4 * i + 2], packed[4 * i + 3]);
        __syncwarp();
#pragma unroll
        for (int kstep = 0; kstep < 4; ++kstep) {
            u32 a[2][4];
#pragma unroll
            for (int half_ = 0; half_ < 2; ++half_) {
                const u16 * tile = slab + (16 * half_ + (lane & 15)) * HEAD_SLAB + 16 * kstep + 8 * (lane >> 4);
                const u32 address = static_cast<u32>(__cvta_generic_to_shared(tile));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(a[half_][0]), "=r"(a[half_][1]), "=r"(a[half_][2]), "=r"(a[half_][3])
                             : "r"(address));
            }
            const int column = 64 * group + 16 * kstep + 2 * (lane & 3);
#pragma unroll
            for (int t = 0; t < TILES; ++t) {  // B fragment: token lane / 4 of the tile, columns 2 (lane % 4) + {0, 1, 8, 9}
                const int token = first_token + 8 * t + (lane >> 2);
                u32 b0 = 0, b1 = 0;
                if (token < tokens) {
                    const half * xt = x_rot + static_cast<size_t>(token) * K + column;
                    b0 = *reinterpret_cast<const u32 *>(xt);
                    b1 = *reinterpret_cast<const u32 *>(xt + 8);
                }
#pragma unroll
                for (int half_ = 0; half_ < 2; ++half_) mma_f16(acc[half_][t], a[half_], b0, b1);
            }
        }
    }

#pragma unroll
    for (int half_ = 0; half_ < 2; ++half_)
#pragma unroll
        for (int t = 0; t < TILES; ++t)
#pragma unroll
            for (int f = 0; f < 4; ++f) {  // C fragment: rows lane / 4 (+ 8 for f >= 2), tokens 2 * (lane % 4) + f % 2
                const int token = first_token + 8 * t + 2 * (lane & 3) + (f & 1);
                const size_t r = first_row + 16 * half_ + (lane >> 2) + 8 * (f >> 1);
                if (token < tokens) logits[static_cast<size_t>(token) * vocab + r] = acc[half_][t][f] * row_scales[r];
            }
}

// ---------------------------------------------------------------------------------------------------------------
// host side

static bool rotation_supported(int64_t K, int order) {
    return (K == 5120 && order == 5) || (K == 6144 && order == 3) || (K == 17408 && order == 17);
}

template <int WIDTH, int ORDER>
void launch_quantize(ggml_backend_cuda_context & ctx, const float * x, const float * rot, uint2 * q, float * stats,
                     int64_t tokens) {
    constexpr int N = WIDTH * ORDER;
    cudaStream_t stream = ctx.stream();
    const float * signs = rot, * small_q = rot + N;
    if (tokens <= SPLIT_TOKENS) {  // the split kernels fill the GPU better than one CTA per token
        ggml_cuda_pool_alloc<float> rotated(ctx.pool(), tokens * N);
        ggml_cuda_pool_alloc<float> column_max(ctx.pool(), tokens * ORDER);
        rotate_columns<WIDTH, ORDER><<<dim3(ORDER, tokens), std::min(1024, WIDTH / 2), 0, stream>>>(
            x, signs, small_q, rotated.get(), column_max.get());
        quantize_rows<WIDTH, ORDER><<<tokens, 1024, 0, stream>>>(rotated.get(), column_max.get(), q, stats);
    } else {
        transform<WIDTH, ORDER><<<tokens, 512, 0, stream>>>(x, signs, small_q, q, stats);
    }
}

// The long-input int8 GEMM through cuBLASLt, with its heuristic's first algorithm per shape: for these shapes that is
// the 256 x 128 tile, which the legacy cublasGemmEx does not pick (it stays on 128 x 64, about half as fast).
struct lt_plan {
    cublasLtMatmulDesc_t   op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t   algo;
};

#define LT_CHECK(x) do { const cublasStatus_t status_ = (x); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { GGML_ABORT("cuBLASLt error %d: %s", (int) status_, #x); } } while (0)

// C (rows x cols, int32, column-major) = A^T B with A the levels (k x rows int8) and B the planes (k x cols int8)
static void lt_int8_gemm(ggml_backend_cuda_context & ctx, int64_t rows, int64_t cols, int64_t k,
                         const void * a, const void * b, int * c) {
    static std::mutex mutex;
    static cublasLtHandle_t handles[GGML_CUDA_MAX_DEVICES] = {};
    static std::map<std::tuple<int, int64_t, int64_t, int64_t>, lt_plan> plans;
    const lt_plan * plan;
    cublasLtHandle_t handle;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!handles[ctx.device]) {
            LT_CHECK(cublasLtCreate(&handles[ctx.device]));
        }
        handle = handles[ctx.device];
        auto & p = plans[{ctx.device, rows, cols, k}];
        if (!p.op) {
            LT_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
            const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
            LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
            LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
            LT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8I, k, rows, k));
            LT_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8I, k, cols, k));
            LT_CHECK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32I, rows, cols, rows));
            cublasLtMatmulPreference_t pref;
            LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
            const size_t no_workspace = 0;
            LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &no_workspace, sizeof(no_workspace)));
            cublasLtMatmulHeuristicResult_t result;
            int found = 0;
            LT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, p.op, p.a, p.b, p.c, p.c, pref, 1, &result, &found));
            GGML_ASSERT(found > 0 && "cuBLASLt has no int8 algorithm for this shape");
            p.algo = result.algo;
            cublasLtMatmulPreferenceDestroy(pref);
        }
        plan = &p;
    }
    const int alpha = 1, beta = 0;
    LT_CHECK(cublasLtMatmul(handle, plan->op, &alpha, a, plan->a, b, plan->b, &beta, c, plan->c, c, plan->c,
                            &plan->algo, nullptr, 0, ctx.stream()));
}

// Per-device state of the pipelined long-input path: two chunk buffers for the decoded levels and the int32 products
// (grown on demand, device-synchronized when they grow), the events that hand them between the two streams, and
// which buffer the next chunk takes. The buffers live outside the pool because the side stream uses them past the
// node that allocated them.
struct pipe_state {
    cudaEvent_t lv_ready[2] = {}, gemm_done[2] = {}, comb_done[2] = {}, fork = nullptr;
    void * levels[2] = {}, * products[2] = {};
    size_t levels_bytes = 0, products_bytes = 0;
    int next = 0;
};

// nullptr when the buffers would have to grow and may not (inside a stream capture): the caller runs the serial path
static pipe_state * pipe_at(ggml_backend_cuda_context & ctx, size_t levels_bytes, size_t products_bytes, bool may_grow) {
    static pipe_state states[GGML_CUDA_MAX_DEVICES];
    pipe_state & ps = states[ctx.device];
    if ((levels_bytes > ps.levels_bytes || products_bytes > ps.products_bytes) && !may_grow) {
        return nullptr;
    }
    if (!ps.fork) {
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ps.lv_ready[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ps.gemm_done[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ps.comb_done[b], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&ps.fork, cudaEventDisableTiming));
    }
    if (levels_bytes > ps.levels_bytes || products_bytes > ps.products_bytes) {
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int b = 0; b < 2; ++b) {
            if (levels_bytes > ps.levels_bytes) {
                if (ps.levels[b]) CUDA_CHECK(cudaFree(ps.levels[b]));
                CUDA_CHECK(cudaMalloc(&ps.levels[b], levels_bytes));
            }
            if (products_bytes > ps.products_bytes) {
                if (ps.products[b]) CUDA_CHECK(cudaFree(ps.products[b]));
                CUDA_CHECK(cudaMalloc(&ps.products[b], products_bytes));
            }
        }
        ps.levels_bytes = std::max(levels_bytes, ps.levels_bytes);
        ps.products_bytes = std::max(products_bytes, ps.products_bytes);
    }
    return &ps;
}

template <typename F>
void launch_mul_mat(ggml_backend_cuda_context & ctx, const u8 * w, int64_t K, int64_t rows, const float * rowscale,
                    const uint2 * q, const float * stats, codebook_t codebook, float * y, int64_t tokens, int prefill_planes) {
    cudaStream_t stream = ctx.stream();
    const int packets_per_row = K / F::columns, groups = rows / 32, groups_per_token = K / 4;
    if (tokens == 1) {
        gemv<F><<<groups, 32 * GEMV_WARPS, 0, stream>>>(w, packets_per_row, rowscale, q, stats, codebook, y);
        return;
    }
    if (tokens <= MMA_TOKENS) {
        const int tile = tokens <= 8 ? 8 : tokens <= 16 ? 16 : (tokens <= 32 || rows % 64) ? 32 : 64;
        const dim3 grid(tile == 64 ? groups / 2 : groups, (tokens + tile - 1) / tile);
        const int n = static_cast<int>(tokens), stride = static_cast<int>(rows);
        switch (tile) {
            case 8:  mma<F, 8><<<grid, 32 * MMA_WARPS, 0, stream>>>(w, packets_per_row, rowscale, q, groups_per_token, n, stats, codebook, y, stride); break;
            case 16: mma<F, 16><<<grid, 32 * MMA_WARPS, 0, stream>>>(w, packets_per_row, rowscale, q, groups_per_token, n, stats, codebook, y, stride); break;
            case 32: mma<F, 32><<<grid, 32 * MMA_WARPS, 0, stream>>>(w, packets_per_row, rowscale, q, groups_per_token, n, stats, codebook, y, stride); break;
            default: mma<F, 64><<<grid, 32 * MMA_WARPS, 0, stream>>>(w, packets_per_row, rowscale, q, groups_per_token, n, stats, codebook, y, stride); break;
        }
        return;
    }
    // long inputs: planes [2 * tokens][K]; the weights in chunks of rows as int8 levels, one int8 GEMM per chunk.
    // prefill_planes == 1 (GGML_MIRAI_PREFILL_PLANES, see ggml_cuda_op_mirai_mul_mat): multiply plane 0 only (half
    // the GEMM work; the activations become per-token int8). The decode and small-batch paths are untouched.
    // GGML_MIRAI_TIMING=1: per-phase GPU time (split / level decode / GEMM / combine) accumulated over calls and
    // printed every 1024 long-input calls (about four 512-token micro-batches). Synchronizes per call: diagnostics.
    static const bool timing = getenv("GGML_MIRAI_TIMING") != nullptr;
    const int64_t plane_cols = prefill_planes * tokens;
    ggml_cuda_pool_alloc<u32> planes(ctx.pool(), 2 * tokens * (K / 4));
    // chunk footprint: the int8 levels (K bytes per row) plus the int32 products (4 bytes per row per plane column),
    // so a chunk is at most LEVELS_MIB of buffers whatever the matrix's K (small K used to give 256 MiB of products)
    const int64_t chunk_rows = std::min<int64_t>(rows, std::max<int64_t>(256, ((static_cast<int64_t>(LEVELS_MIB) << 20) / (K + plane_cols * 4)) / 256 * 256));
    // GGML_MIRAI_PIPELINE=1 (experiment; default off, measured 0 to -2.5% for +120-220 MiB on a 4070 because the int8
    // GEMM runs at the tensor peak and leaves the SMs no room for the decode to co-run): the level decodes run on one
    // side stream and the combines on another against the GEMMs on the main stream, through two persistent chunk
    // buffers (levels, products) handed over by events. A decode waits only for the GEMM that last read its buffer, so
    // the next node's first decode runs during this node's last GEMM (the decode stream must not queue behind the
    // combines, hence the third stream); a combine waits for its GEMM and runs during the next one. The node joins both
    // side streams before it returns, so y is complete for the main stream and the pool may reuse `planes`.
    static const bool pipeline = env_tokens("GGML_MIRAI_PIPELINE", 0) != 0;
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capturing));
    pipe_state * pipe_ptr = pipeline && !timing
        ? pipe_at(ctx, static_cast<size_t>(chunk_rows) * K, static_cast<size_t>(plane_cols) * chunk_rows * sizeof(int), capturing == cudaStreamCaptureStatusNone)
        : nullptr;
    if (pipe_ptr) {
        pipe_state & ps = *pipe_ptr;
        split_planes<<<dim3((groups_per_token + 255) / 256, tokens), 256, 0, stream>>>(q, groups_per_token, tokens, planes.get());
        cudaStream_t dec = ctx.stream(ctx.device, 1), cmb = ctx.stream(ctx.device, 2);
        if (capturing != cudaStreamCaptureStatusNone) {
            // fork the side streams from this capture; earlier work on both buffers is behind the main stream's tail
            CUDA_CHECK(cudaEventRecord(ps.fork, stream));
            CUDA_CHECK(cudaStreamWaitEvent(dec, ps.fork, 0));
            CUDA_CHECK(cudaStreamWaitEvent(cmb, ps.fork, 0));
            for (int b = 0; b < 2; ++b) {
                CUDA_CHECK(cudaEventRecord(ps.gemm_done[b], stream));
                CUDA_CHECK(cudaEventRecord(ps.comb_done[b], stream));
            }
        }
        const int n_chunks = static_cast<int>((rows + chunk_rows - 1) / chunk_rows);
        auto decode = [&](int chunk) {
            const int b = (ps.next + chunk) & 1;
            CUDA_CHECK(cudaStreamWaitEvent(dec, ps.gemm_done[b], 0));  // the GEMM that last read levels[b]
            const int64_t first = static_cast<int64_t>(chunk) * chunk_rows, n_rows = std::min(chunk_rows, rows - first);
            levels<F><<<n_rows / 32, 32 * GEMV_WARPS, 0, dec>>>(w, packets_per_row, first / 32, static_cast<u32 *>(ps.levels[b]), groups_per_token);
            CUDA_CHECK(cudaEventRecord(ps.lv_ready[b], dec));
        };
        decode(0);
        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const int b = (ps.next + chunk) & 1;
            if (chunk + 1 < n_chunks) decode(chunk + 1);
            const int64_t first = static_cast<int64_t>(chunk) * chunk_rows, n_rows = std::min(chunk_rows, rows - first);
            CUDA_CHECK(cudaStreamWaitEvent(stream, ps.lv_ready[b], 0));
            CUDA_CHECK(cudaStreamWaitEvent(stream, ps.comb_done[b], 0));  // the combine that last read products[b]
            lt_int8_gemm(ctx, n_rows, plane_cols, K, ps.levels[b], planes.get(), static_cast<int *>(ps.products[b]));
            CUDA_CHECK(cudaEventRecord(ps.gemm_done[b], stream));
            CUDA_CHECK(cudaStreamWaitEvent(cmb, ps.gemm_done[b], 0));
            prefill_output<<<dim3((n_rows + 255) / 256, tokens), 256, 0, cmb>>>(
                static_cast<int *>(ps.products[b]), tokens, n_rows, rowscale + first, stats, codebook, y + first, rows, prefill_planes == 1);
            CUDA_CHECK(cudaEventRecord(ps.comb_done[b], cmb));
        }
        // join: the combines are in order on their stream, so the last one covers them all; the decodes of this node
        // all precede its GEMMs, which the main stream already ordered
        CUDA_CHECK(cudaStreamWaitEvent(stream, ps.comb_done[(ps.next + n_chunks - 1) & 1], 0));
        ps.next = (ps.next + n_chunks) & 1;
        return;
    }
    cudaEvent_t ev[4];
    if (timing) { for (auto & e : ev) CUDA_CHECK(cudaEventCreate(&e)); CUDA_CHECK(cudaEventRecord(ev[0], stream)); }
    split_planes<<<dim3((groups_per_token + 255) / 256, tokens), 256, 0, stream>>>(q, groups_per_token, tokens, planes.get());
    float t_split = 0.0f, t_levels = 0.0f, t_gemm = 0.0f, t_out = 0.0f;
    if (timing) { CUDA_CHECK(cudaEventRecord(ev[1], stream)); CUDA_CHECK(cudaEventSynchronize(ev[1])); CUDA_CHECK(cudaEventElapsedTime(&t_split, ev[0], ev[1])); }
    ggml_cuda_pool_alloc<u32> levels_buf(ctx.pool(), chunk_rows * (K / 4));
    ggml_cuda_pool_alloc<int> products(ctx.pool(), plane_cols * chunk_rows);
    for (int64_t first = 0; first < rows; first += chunk_rows) {
        const int64_t n_rows = std::min(chunk_rows, rows - first);
        if (timing) CUDA_CHECK(cudaEventRecord(ev[0], stream));
        levels<F><<<n_rows / 32, 32 * GEMV_WARPS, 0, stream>>>(w, packets_per_row, first / 32, levels_buf.get(), groups_per_token);
        if (timing) CUDA_CHECK(cudaEventRecord(ev[1], stream));
        // column-major: C (n_rows x plane_cols) = levels^T (n_rows x K) * planes (K x plane_cols); plane 0 comes first
        lt_int8_gemm(ctx, n_rows, plane_cols, K, levels_buf.get(), planes.get(), products.get());
        if (timing) CUDA_CHECK(cudaEventRecord(ev[2], stream));
        prefill_output<<<dim3((n_rows + 255) / 256, tokens), 256, 0, stream>>>(
            products.get(), tokens, n_rows, rowscale + first, stats, codebook, y + first, rows, prefill_planes == 1);
        if (timing) {
            CUDA_CHECK(cudaEventRecord(ev[3], stream)); CUDA_CHECK(cudaEventSynchronize(ev[3]));
            float a, b, c;
            CUDA_CHECK(cudaEventElapsedTime(&a, ev[0], ev[1])); CUDA_CHECK(cudaEventElapsedTime(&b, ev[1], ev[2])); CUDA_CHECK(cudaEventElapsedTime(&c, ev[2], ev[3]));
            t_levels += a; t_gemm += b; t_out += c;
        }
    }
    if (timing) {
        for (auto & e : ev) CUDA_CHECK(cudaEventDestroy(e));
        static double s_split = 0, s_levels = 0, s_gemm = 0, s_out = 0; static int64_t s_calls = 0, s_params = 0, s_tokens = 0;
        s_split += t_split; s_levels += t_levels; s_gemm += t_gemm; s_out += t_out; s_calls++; s_params += K * rows; s_tokens += tokens;
        if (s_calls % 1024 == 0) {
            const double total = s_split + s_levels + s_gemm + s_out;
            GGML_LOG_INFO("mirai prefill timing over %lld calls (%lld params, %lld token-rows, planes %d): split %.1f%% levels %.1f%% gemm %.1f%% combine %.1f%%; "
                          "levels %.0f Gparam/s, gemm %.1f ms per 512 tokens per 27B\n",
                          (long long) s_calls, (long long) s_params, (long long) s_tokens, prefill_planes,
                          100 * s_split / total, 100 * s_levels / total, 100 * s_gemm / total, 100 * s_out / total,
                          s_params / (s_levels * 1e6), s_gemm * (512.0 / (double) s_tokens * s_calls) * (27e9 / (double) s_params * s_calls) / s_calls);
            s_split = s_levels = s_gemm = s_out = 0; s_calls = s_params = s_tokens = 0;
        }
    }
}

} // namespace

bool ggml_cuda_mirai_supports_op(const ggml_tensor * op) {
    if (op->op == GGML_OP_MIRAI_QUANTIZE) {
        if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
            return false;
        }
        return ggml_get_op_params_i32(op, 0) ? op->src[0]->ne[0] % 256 == 0
                                             : rotation_supported(op->src[0]->ne[0], ggml_get_op_params_i32(op, 1));
    }
    const ggml_tensor * w = op->src[0];
    const int64_t K = w->ne[0], rows = w->ne[1];
    switch (w->type) {
        case GGML_TYPE_MS_I3:
            return K % 128 == 0 && rows % 256 == 0;
        case GGML_TYPE_MS_V4T8:
        case GGML_TYPE_MS_V2T4:
            return K % 512 == 0 && rows % 32 == 0;  // mma n64 pairs warps: packets per row % 8 == 0
        case GGML_TYPE_MS_V2T6:
            return K % 1024 == 0 && rows % 32 == 0;
        default:
            return false;
    }
}

void ggml_cuda_op_mirai_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const float * rot = static_cast<const float *>(dst->src[1]->data);
    const int64_t K = x->ne[0], tokens = ggml_nrows(x);
    const float * x_d = static_cast<const float *>(x->data);
    if (ggml_get_op_params_i32(dst, 0)) {
        head_input<<<tokens, 256, 0, ctx.stream()>>>(x_d, rot, static_cast<half *>(dst->data), K);
        return;
    }
    uint2 * q = static_cast<uint2 *>(dst->data);
    float * stats = reinterpret_cast<float *>(q + tokens * (K / 4));
    switch (K) {
        case 5120:  launch_quantize<1024, 5>(ctx, x_d, rot, q, stats, tokens); break;
        case 6144:  launch_quantize<2048, 3>(ctx, x_d, rot, q, stats, tokens); break;
        case 17408: launch_quantize<1024, 17>(ctx, x_d, rot, q, stats, tokens); break;
        default: GGML_ABORT("Mirai S: unsupported rotation width %lld", (long long) K);
    }
}

void ggml_cuda_op_mirai_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * xq = dst->src[1];
    const float * rowscale = static_cast<const float *>(dst->src[2]->data);
    const int64_t K = w->ne[0], rows = w->ne[1], tokens = xq->ne[1];
    const u8 * w_d = static_cast<const u8 *>(w->data);
    float * y = static_cast<float *>(dst->data);
    cudaStream_t stream = ctx.stream();

    if (w->type == GGML_TYPE_MS_I3) {
        const half * x_rot = static_cast<const half *>(xq->data);
        const float * ladder = static_cast<const float *>(dst->src[3]->data);
        const int tile = tokens <= 8 ? 8 : tokens <= 16 ? 16 : 32;
        const dim3 grid(rows / (32 * HEAD_WARPS), (tokens + tile - 1) / tile);
        const int n = static_cast<int>(tokens), k = static_cast<int>(K), v = static_cast<int>(rows);
        switch (tile) {
            case 8:  head_mma<8><<<grid, 32 * HEAD_WARPS, 0, stream>>>(x_rot, n, w_d, k, rowscale, ladder, y, v); break;
            case 16: head_mma<16><<<grid, 32 * HEAD_WARPS, 0, stream>>>(x_rot, n, w_d, k, rowscale, ladder, y, v); break;
            default: head_mma<32><<<grid, 32 * HEAD_WARPS, 0, stream>>>(x_rot, n, w_d, k, rowscale, ladder, y, v); break;
        }
        return;
    }

    codebook_t codebook;
    memcpy(codebook.c, dst->op_params, sizeof(codebook.c));
    const uint2 * q = static_cast<const uint2 *>(xq->data);
    const float * stats = reinterpret_cast<const float *>(q + tokens * (K / 4));
    // GGML_MIRAI_PREFILL_PLANES: unset/2 = both activation planes (exact); 1 = plane 0 only for every long-input
    // matmul; ffn = plane 0 only for the ffn_* matmuls (~75% of the prefill GEMM work), both planes for the
    // attention projections (their K/V persist in the cache) and ssm_out. Experimental, measured by KL.
    static const int plane_mode = [] {
        const char * s = getenv("GGML_MIRAI_PREFILL_PLANES");
        return s == nullptr ? 0 : strcmp(s, "1") == 0 ? 1 : strcmp(s, "ffn") == 0 ? 2 : 0;
    }();
    const int planes = plane_mode == 1 ? 1 : (plane_mode == 2 && strstr(w->name, "ffn_") != nullptr) ? 1 : 2;
    switch (w->type) {
        case GGML_TYPE_MS_V4T8: launch_mul_mat<fmt_v4t8>(ctx, w_d, K, rows, rowscale, q, stats, codebook, y, tokens, planes); break;
        case GGML_TYPE_MS_V2T4: launch_mul_mat<fmt_v2t4>(ctx, w_d, K, rows, rowscale, q, stats, codebook, y, tokens, planes); break;
        case GGML_TYPE_MS_V2T6: launch_mul_mat<fmt_v2t6>(ctx, w_d, K, rows, rowscale, q, stats, codebook, y, tokens, planes); break;
        default: GGML_ABORT("Mirai S: not a trellis type");
    }
}

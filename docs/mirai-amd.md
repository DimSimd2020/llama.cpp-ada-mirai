# Mirai S on AMD HIP

This fork adds HIP kernels for `GGML_OP_MIRAI_QUANTIZE` and `GGML_OP_MIRAI_MUL_MAT`.
The original Mirai S GGUF stays unchanged. All three trellis formats (MS_V4T8,
MS_V2T4, MS_V2T6) and the MS_I3 output head are supported.

The rotation and two-plane quantization follow the CPU reference. The matrix
kernel decodes trellis packets on the GPU and uses AMD integer dot products.
Single-token decode and four-token verification tiles share the same math. Batches of at least 16 tokens decode row chunks and use exact int8/int32 hipBLAS GEMM for both activation planes. Trellis states are computed from the bit window without a serial replay dependency. The MS_I3
head rounds its ladder and weights to FP16 before multiplication, as the CPU
reference does. The HIP implementation does not use the approximate one-plane
prefill option or NVIDIA tensor-core instructions.

On Windows, build with the ROCm Clang compiler from a Visual Studio developer
prompt, with a complete HIP SDK available through `CMAKE_PREFIX_PATH`:

```bat
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_HIP_NO_VMM=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_C_COMPILER=clang.exe -DCMAKE_CXX_COMPILER=clang++.exe -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-hip --parallel 6 --target llama-server test-backend-ops
```

The CPU comparison tests are in the existing backend test executable:

```bat
test-backend-ops.exe -o MIRAI_MUL_MAT -b ROCm0
```

They cover all four weight formats, the three rotation shapes used by Qwen3.8
27B, token batches of 1, 2, 3, 4, 7, 17 and 64, dense rotations, and a zero input.

Set `HIP_VISIBLE_DEVICES` to the discrete GPU's physical HIP ordinal before
starting the process. After filtering, its llama.cpp device name is `ROCm0`.
The RX 7900 XT is `gfx1100`. Device enumeration can differ between computers;
use `hipInfo` and `llama-server --list-devices` to select it.

```bat
llama-server.exe -m Qwen3.8-27B-S-mirai.gguf -ngl 99 --device ROCm0 -fa on -c 8192 -np 1 --reasoning off
```

The published control vector works with the existing `--cvec-mode project`
implementation. On Windows, use a relative vector path with
`--control-vector-scaled`: the parser treats `:` as the scale separator.

The Vulkan backend does not implement these operations in this fork. NVIDIA
performance measurements do not describe the HIP kernels.

Validated on Windows with ROCm SDK 10.0.0 and an RX 7900 XT (gfx1100):
96/96 Mirai backend tests passed against the CPU reference. The model served
through the OpenAI-compatible API with an 8192-token context, flash attention,
one slot, and the published control vector. New-text and copying throughput
are measured separately because lookup drafting can reuse known text.

The local launcher uses these additional settings:

```bat
--backend-sampling --spec-type ngram-mod,draft-mtp --spec-lookup-n-max 32 --spec-draft-n-max 2 --spec-draft-n-max-tail 2 --spec-draft-window 8192 -ctkd q8_0 -ctvd q8_0 -b 1024 -ub 512
```

These are one-machine measurements, not a performance guarantee. Larger MTP
drafts, the optional DFlash drafter, and a large decoded-weight cache were
slower on this setup and are not part of the selected configuration.

HIP TOP_K and ARGSORT now use hipCUB radix sorting for rows larger than 1024
values. This covers the model's 248320-token vocabulary and allows top-k/top-p
backend sampling without the former unsupported-operation fallback. All 615
existing TOP_K/ARGSORT comparison tests passed, including ties, row batches,
non-power-of-two sizes, and large rows. Keep the reasoning budget unlimited
when using backend sampling; this server disables backend sampling if a
reasoning-budget sampler is active, even when thinking is off.

The next optimization specializes complete one-, two-, three- and four-token
tiles, avoiding per-token bounds checks in the inner matrix loops. Partial
four-token tiles still check bounds. The output head decodes each weight pair
once per tile and uses packed FP16 multiplication with FP32 dot-product
accumulation. Rotation uses two warp-local Hadamard transforms with one padded
shared-memory transpose, preserving the butterfly order and both activation
planes.

On the same RX 7900 XT, the first 52/192-token request after a server restart
generated 45.57 tokens/s and took 4.61 seconds. A three-prompt comparison
(DNS explanation, fictional story, seasons explanation; 256 output tokens each)
improved from 35.43/30.23/35.10 to 48.22/40.04/48.40 tokens/s. Each prompt was
new to its respective process, and the generated texts matched. These are
new-text measurements, not repeats of a response learned by lookup drafting.
All 711 CPU comparison tests passed: 96 Mirai cases and the existing 615
TOP_K/ARGSORT cases. The main GGUF and control vector are unchanged.

The existing performance harness now also covers the real 248320-row output
head and 5120/17408 matrix shapes. On Windows, large repeated-node HIP graphs
can exhaust the test process's stack; disable graphs for this isolated kernel
benchmark only:

```bat
set GGML_CUDA_DISABLE_GRAPHS=1
test-backend-ops.exe perf -o MIRAI_MUL_MAT -b ROCm0
```

Keep HIP graphs enabled in the serving process. End-to-end measurements above
use graphs, MTP draft size 2, backend sampling and lookup drafting.

# llama.cpp for Mirai S on 12 GB Ada

This is [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) with two sets of changes on top:

1. **The serving patches** from [bonsai-ada-surgery](https://github.com/professorpalmer/bonsai-ada-surgery) (also
   open as PrismML PRs #319-#323): the tiered KV cache (`--kv-vram-cells`: a VRAM head and a pinned-host tail in one
   CUDA address range, copy-engine staging), MTP drafting at every depth (`--spec-draft-window`,
   `--spec-draft-n-max-tail`), harness-proofing (`--reasoning-effort-allow/-fallback`, `--reasoning-max-tokens-floor`),
   batch-invariant kernels (`GGML_CUDA_BATCH_INVARIANT`), per-op GPU timing (`GGML_CUDA_OP_TIMING`).
2. **Mirai S** ([alesha-pro/Qwen3.8-27B-S-mirai-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-27B-S-mirai-GGUF)):
   the trellis codec ported from [alesha-pro/llama.cpp-mirai-s](https://github.com/alesha-pro/llama.cpp-mirai-s)
   (ggml types 90-93 with CPU and CUDA kernels, model-wide rotation tensors and per-row scales, the split attention
   gate, the graph hook), checked greedy token-for-token against that fork; then the prefill work done here:
   `GGML_MIRAI_PREFILL_PLANES=ffn` (one int8 activation plane for the FFN matmuls of prompt-sized batches, +18%
   prefill at KL 0.00028), the packed 1-bit KQ mask (`--kq-mask-packed`, read natively by the tensor-core attention
   kernel), and the level-decode chunk bounded by its buffer footprint (`GGML_MIRAI_LEVELS_MIB`).

Build like any llama.cpp with `-DGGML_CUDA=ON`; the product launcher, measurements and receipts live in
[mirai-s-ada](https://github.com/professorpalmer/mirai-s-ada). Upstream llama.cpp's README is `README.md`.

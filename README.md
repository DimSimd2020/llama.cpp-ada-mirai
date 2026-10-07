# llama.cpp for Mirai S on 12 GB Ada

This is [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) with two sets of changes on top:

1. **The serving patches** from [bonsai-ada-surgery](https://github.com/professorpalmer/bonsai-ada-surgery) (also
   open as PrismML PRs #319-#323): the tiered KV cache (`--kv-vram-cells`: a VRAM head and a pinned-host tail in one
   CUDA address range, copy-engine staging), MTP drafting at every depth (`--spec-draft-window`,
   `--spec-draft-n-max-tail`), harness-proofing (`--reasoning-effort-allow/-fallback`, `--reasoning-max-tokens-floor`),
   batch-invariant kernels (`GGML_CUDA_BATCH_INVARIANT`), per-op GPU timing (`GGML_CUDA_OP_TIMING`).
2. **Mirai S** (Mirai Labs' Qwen3.8-27B-S, in alesha-pro's GGUF conversion
   [alesha-pro/Qwen3.8-27B-S-mirai-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-27B-S-mirai-GGUF)):
   Mirai Labs' trellis codec, as alesha-pro ported it to ggml in [alesha-pro/llama.cpp-mirai-s](https://github.com/alesha-pro/llama.cpp-mirai-s)
   (ggml types 90-93 with CPU and CUDA kernels, model-wide rotation tensors and per-row scales, the split attention
   gate, the graph hook), checked greedy token-for-token against that fork; then the prefill work done here:
   `GGML_MIRAI_PREFILL_PLANES=ffn` (one int8 activation plane for the FFN matmuls of prompt-sized batches, +18%
   prefill at KL 0.00028), the packed 1-bit KQ mask (`--kq-mask-packed`, read natively by the tensor-core attention
   kernel), and the level-decode chunk bounded by its buffer footprint (`GGML_MIRAI_LEVELS_MIB`).

3. **Control-vector projection** (`--cvec-mode project`, from alesha-pro): llama.cpp's control vector applied as
   `h -= |d| (h.v) v` after every layer in its range instead of `h += d`. A unit direction at scale 1.0 is removed
   from the residual stream. With the refusal direction published next to the weights this is a run-time
   abliteration for trellis codes that cannot be edited; layers without a direction in the file are left alone, and
   without the flag nothing changes (greedy identity re-checked).

Build like any llama.cpp with `-DGGML_CUDA=ON`; the product launcher, measurements and receipts live in
[mirai-s-ada](https://github.com/professorpalmer/mirai-s-ada). The fork's original README is `README-upstream.md`;
licenses: MIT throughout (the ggml authors, PrismML, alesha-pro; notices preserved). The model and the Mirai S codec
are Mirai Labs' (base model Qwen3.8-27B by Qwen). The planar activation layout and the batch-invariant mode are
sudoingX's ([PrismML PR #218](https://github.com/PrismML-Eng/llama.cpp/pull/218)), kept with their authorship. Not
affiliated with any of them.

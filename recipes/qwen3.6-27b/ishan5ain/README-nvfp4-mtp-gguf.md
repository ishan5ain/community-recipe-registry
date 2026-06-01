# qwen3.6-27b-nvfp4-mtp-llamacpp-ishan5ain

**Qwen3.6-27B NVFP4 + MTP via llama.cpp** on a single DGX Spark GB10
(128 GB unified memory, Blackwell SM 12.1).

## Model

| Property | Value |
|----------|-------|
| Source | [nilayparikh/Qwen3.6-27B-Text-NVFP4-MTP-GGUF](https://huggingface.co/nilayparikh/Qwen3.6-27B-Text-NVFP4-MTP-GGUF) |
| Base | Qwen/Qwen3.6-27B |
| Quant | NVFP4 (modelopt, group_size=16) |
| Weights | ~19.7 GB (GGUF) |
| Spec Decode | MTP-3 (built-in, no external draft model) |
| Calibration | neuralmagic/calibration (20 samples) |
| License | Apache 2.0 |

This is a GGUF conversion of [sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP)
— a modelopt-quantized NVFP4 variant with the MTP head preserved in BF16
(unlike `compressed-tensors` exports which drop it).

## Quick Start

```bash
# Base recipe (short context, max throughput)
sparkrun run @community/qwen3.6-27b-nvfp4-mtp-llamacpp-ishan5ain --solo

# Long context variant (192K with YaRN)
sparkrun run @community/qwen3.6-27b-nvfp4-mtp-longctx-llamacpp-ishan5ain --solo
```

## VRAM Estimation

Base recipe (32K context):

| Component | Size |
|-----------|------|
| Model weights (GGUF NVFP4 + BF16 MTP) | ~19.7 GB |
| KV cache (F16, 32K ctx) | ~2 GB |
| Activation + overhead | ~4 GB |
| **Total** | **~26 GB** |
| Available (80% of 128 GB) | ~96.8 GB |
| Headroom | ~70 GB |

Long context (192K context):

| Component | Size |
|-----------|------|
| Model weights | ~19.7 GB |
| KV cache (F16, 192K ctx) | ~24 GB |
| Activation + overhead | ~6 GB |
| **Total** | **~50 GB** |
| Available (80% of 128 GB) | ~96.8 GB |
| Headroom | ~47 GB |

## Why This Combination Matters

This is the **first community GGUF that bundles NVFP4 weights + MTP heads** in a
single file for llama.cpp. This matters because:

1. **No external draft model** — MTP uses Qwen3.6's built-in multi-token prediction
   heads, unlike DFlash which needs a separate draft model (~1.7 GB download,
   ~1.6 GiB GPU memory).
2. **llama.cpp avoids vLLM's NVFP4 issues on DGX Spark** — no TMEM requirement,
   no FlashInfer autotune, no driver lock (vLLM needs 595.58+).
3. **Smaller memory footprint than Q4_K_M** — NVFP4 at ~14 GB equivalent weights
   vs Q4_K_M at ~16 GB, freeing room for longer context or concurrent sequences.

## Comparison with @ishan5ain Recipes

See the companion analysis in `ANALYSIS.md` for detailed comparison with all
seven recipes tested in `@recipes/qwen3.6-27b/ishan5ain/LEARNINGS.md`.

## Caveats

- **Requires llama.cpp with MTP support** (`-mtp-n` flag). The stock spark-arena
  container should include this. Verify with `llama-server --help | grep mtp`.
- **NVFP4 GGUF is a new format** — ensure your llama.cpp build was compiled with
  NVFP4 tensor type support (default on with GGML_CUDA=ON for SM 12.x).
- **MTP acceptance rate TBD** on GB10 for this specific GGUF conversion. The
  source model (sakamakismile) achieved 64-91% MTP acceptance via vLLM, but
  llama.cpp's MTP implementation may differ.
- **Speedhack fork may be faster** — the model author's 40 tok/s claim was
  achieved with the phuongncn speedhack fork, not stock llama.cpp.

## Credits

Model by [@nilayparikh](https://huggingface.co/nilayparikh) — Qwen3.6-27B-Text-NVFP4-MTP-GGUF.
Recipe authored by @ishan5ain.

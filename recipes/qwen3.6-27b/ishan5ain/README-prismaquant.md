# qwen3.6-27b-prismaquant-5.5bit-mtp3-vllm-ishan5ain

**Qwen3.6-27B PrismaQuant 5.5 bpp + MTP** on a single DGX Spark GB10
(128 GB unified memory, Blackwell SM 12.1).

PrismaQuant is a per-Linear sensitivity-driven mixed-precision quantization.
Unlike uniform NVFP4 (4 bits everywhere), this artifact allocates each Linear
module its own format — NVFP4, MXFP8, or BF16 — chosen under a 5.5-bit
budget to minimize predicted Δloss. The 5.5 bpp target was selected as the
Pareto knee of the Δloss-vs-bpp curve: the point where further bit-budget
buys less marginal Δloss reduction than the bits already spent.

**Precision mix summary:**

| Format | Count | Where |
|--------|-------|-------|
| NVFP4 (4-bit, gs=16) | 349 | Bulk dense MLPs, medium-sensitivity attention, most visual Linears |
| MXFP8 (8-bit, E4M3) | 35 | High-sensitivity dense Linears too risky at 4-bit |
| BF16 | 112+352 | Highest-sensitivity Linears, norms, biases, embed |

## Quick Start

```bash
sparkrun run @community/qwen3.6-27b-prismaquant-5.5bit-mtp3-vllm-ishan5ain --solo
```

Or from the local file:

```bash
sparkrun run recipes/qwen3.6-27b/ishan5ain/qwen3.6-27b-prismaquant-5.5bit-mtp3-vllm-ishan5ain.yaml --solo
```

## VRAM Estimation

| Component | Size |
|-----------|------|
| Model weights (mixed NVFP4+MXFP8+BF16) | ~19 GB |
| KV cache (FP8, 32K ctx) | ~4 GB |
| Activation memory + overhead | ~4 GB |
| **Total** | **~27 GB** |
| Available @ 90% of 128 GB | ~115 GB |
| Headroom | **~88 GB** |

## MTP Speculative Decoding

This recipe enables MTP at n=3 tokens, which the PrismaQuant author measured as
the optimum on DGX Spark (n=2 leaves ~10% tok/s on the table, n=4 regresses).
The MTP head is quantized at the same mixed precision as the body — no BF16
fallback.

## Customisation

### Adjust context length

```bash
sparkrun run @community/qwen3.6-27b-prismaquant-5.5bit-mtp3-vllm-ishan5ain \
  --solo \
  -o max_model_len=65536
```

With ~88 GB headroom, you can push context well beyond 32K.

### Adjust GPU memory

```bash
sparkrun run @community/qwen3.6-27b-prismaquant-5.5bit-mtp3-vllm-ishan5ain \
  --solo \
  -o gpu_memory_utilization=0.95
```

### Disable MTP for lower latency (single-turn)

```bash
sparkrun run @community/qwen3.6-27b-prismaquant-5.5bit-mtp3-vllm-ishan5ain \
  --solo \
  -o speculative_config='{"method":"mtp","num_speculative_tokens":0}'
```

## Known Issues & Limitations

### 1. vLLM only

This precision mix (NVFP4+MXFP8+BF16) has no transformers-native runtime path.
vLLM 0.11+ with `compressed-tensors` support is required.

### 2. lm_head stays BF16

vLLM's `ParallelLMHead` does not register NVFP4/MXFP4 compressed-tensors
schemes, so `lm_head` is force-preserved at BF16 (~770 MB overhead). This
is a vLLM runtime limitation, not a quantization design issue.

### 3. Multimodal support

Vision inputs work via vLLM's standard `image-text-to-text` chat API —
no special flags needed.

### 4. NVFP4 GEMM backend

The recipe sets `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` explicitly.
This is auto-detected by default but pinned for deterministic behavior.

## Author

[@ishan5ain](https://github.com/ishan5ain)

Artifact by [@RobTand](https://github.com/RobTand) / [PrismaQuant](https://github.com/RobTand/prismaquant)

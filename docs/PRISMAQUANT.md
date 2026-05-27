# PrismaQuant — Mixed-Precision LLM Quantization

> **Source:** [github.com/RobTand/prismaquant](https://github.com/RobTand/prismaquant)
> **Author:** Robert Tand
> **Summary generated:** 2026-05-26

---

## What is PrismaQuant?

A mixed-precision quantization framework for large language models (LLMs). Given a model and a target size (e.g. 4.75 bits per parameter), it decides **which quantization format each linear layer gets** — NVFP4, MXFP8, BF16, INT4, etc. — so the compressed model loses as little quality as possible.

The output is a standard `compressed-tensors` checkpoint that **vLLM serves natively** — no custom kernels, no forked runtime, no patches.

```bash
vllm serve $WORK_DIR/exported --quantization compressed-tensors
```

---

## Headline Results

| Model | Result |
|---|---|
| **Qwen3.6-27B (PrismaSCOUT)** | **11% smaller, 68% lower KL** than prior ship — 20.17 GB vs 22.67 GB at 5.31 bpp |
| **Qwen3.6-35B-A3B @ 4.75 bpp** | **Wins 8 of 9 zero-shot metrics** vs uniform NVFP4 (RedHatAI baseline). Mean drop vs BF16: −0.56 pp (PrismaQuant) vs −2.21 pp (uniform NVFP4). Ships 2 GB smaller. |

Public HuggingFace artifact: [`rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm) (DOI `10.57967/hf/8656`).

---

## How It Works

Quantization decomposes into two questions:

1. **How should each linear be rounded?** — GPTQ, AutoRound, scale sweeps, activation clipping (well-studied).
2. **How many bits should each linear get?** — **This is where PrismaQuant operates.**

### Classical approach (v1 pipeline)

$$\Delta \text{loss} \approx \frac{1}{2} \cdot H_{\text{trace}} \cdot \text{MSE}_W$$

- `H_trace` = empirical Fisher diagonal trace (one calibration pass — measures loss sensitivity to each weight)
- `MSE_W` = per-format round-trip error on actual weights
- Plug into a **multi-choice knapsack** solved via DP in seconds

### The core insight: PrismaSCOUT

The problem: per-linear sensitivity scores are **biased estimators of joint quantization error**. When you flip many linears at once, the additive surrogate systematically overshoots measured KL by 30–50%.

**PrismaSCOUT's slogan:** *Surrogates generate, real KL selects.*

A multi-level cost cascade:

| Level | What it does | Time |
|---|---|---|
| **L1** | Fisher-weighted MSE per `(Linear, format)`. Solve additive DP. | CPU-seconds |
| **L2** | Install activation hooks under L1 assignment, cache activations, re-measure MSE under *perturbed* activation distribution, re-solve DP. Iterate to convergence (~3 passes). | ~3 calibration passes |
| **L3** | Select uncertain linears, measure *paired BF16/candidate end-KL* on each, solve frozen DP at budget. | Held-out split |

Then a **validated-frontier kneedle** runs the cascade at multiple budgets, filters by Pareto-dominance, and picks the elbow on measured `(bpp, KL)`. A **monotone coordinate-descent polish** perturbs locally and only accepts flips that strictly improve real KL.

### Most recent refinement: production-faithful polish

Evaluates flips on the *actual export-aligned weight path* (joint NVFP4 sibling-coherent input global scales, GPTQ, scale sweep, calibrated activation clip) instead of round-trip proxy weights. Polish move units are **Block-CLADO decision units** — fused-sibling linears (`q/k/v`, `gate/up/down`) grouped into atomic flip targets. A delta-quantize `WeightSession` swaps one decision unit in place per trial instead of cloning the model, making polish tractable on a 27B model under a 121 GB budget.

---

## Pipeline

```
sensitivity_probe ──► probe.pkl       (Fisher H_trace per Linear)
         │
measure_quant_cost ─► cost.pkl        (per-(Linear, format) MSE — L1)
         │
iterate_perturbed_allocation ─► validated_frontier.json   (L2+L3, validated kneedle)
         │
polish_from_assignment ─► polished.json                    (production-faithful polish)
         │
export_native_compressed ─► exported/                      (compressed-tensors checkpoint)
         │
validate_native_export   ─► vLLM forward + greedy decode + perplexity gate
```

For models >200B (MiniMax M2.7 @ 228B, DeepSeek-V4-Flash @ 671B), a **streaming layer-by-layer path** keeps peak memory bounded — no full-model load required.

---

## Supported Formats

| Family | Formats |
|---|---|
| NVIDIA microscaling | NVFP4, NVFP4A16 |
| MX (Open Compute) | MXFP4, MXFP6_E3M2, MXFP6_E2M3, MXFP8, MXFP8A16 |
| Integer | INT8_W8A16, INT4_W4A16_g128 |
| Native passthrough | BF16, FP8_SOURCE (byte-exact preservation) |

The allocator is **constraint-aware**: it never picks a format vLLM can't serve.

Hardware support:

| | Blackwell (SM100+) | Ampere/Ada | vLLM serving today |
|---|---|---|---|
| NVFP4 | ✓ (CUTLASS) | Marlin emu | ✓ |
| MXFP4 | ✓ (CUTLASS) | Marlin emu | ✓ |
| MXFP6 | ✓ (native) | — | ✗ (kernel pending) |
| MXFP8 / FP8 | ✓ (CUTLASS) | ✓ | ✓ |
| INT4 / INT8 | all NV | all NV | ✓ (Marlin) |

---

## Code Architecture

The codebase (~70+ Python modules in `prismaquant/`) is well-engineered:

- **`format_registry.py`** — extensible `@register_format` decorator. Each `FormatSpec` carries its own RTN quantize-dequantize, AutoRound config, effective bit calculation, minimum SM capability.
- **`allocator.py`** — multi-choice knapsack DP with fusion-projections (siblings promoted to highest format). Auto-Pareto knee via Kneedle.
- **`decision_units.py`** — Block-CLADO fused-sibling grouping.
- **`weight_session.py`** — delta-quantize swap-in-place for tractable polish on large models.
- **`production_weight_cache.py`**, **`perturbed_x_cache.py`**, **`layer_streaming.py`** — streaming path for >200B models.
- **`kl_measurement.py`**, **`kl_fisher.py`**, **`kl_sensitivity_probe.py`** — real KL measurement infrastructure.

Agent rules (`AGENTS.md`) enforce: GPU-bound by default, reuse cache system, measure on same calibration contract, report bpp over quantizable parameters only.

---

## Related Work & Rejected Detours

The README and paper are unusually frank about what didn't work:

- **CLADO** (Deng et al. 2023) — foundational pairwise coupling formulation; PrismaQuant's Block-CLADO builds on it
- **HAWQ-V1/V2/V3** (Dong et al. 2019–2021) — Fisher-based mixed-precision allocation lineage
- **CoopQ** (Zhao et al. 2025) — cooperative-game view of allocation
- **GPTQ**, **AutoRound** — rounding algorithms that compose underneath the allocator
- Rejected: Lagrangian λ-bisection, sandwich proximal recalibration, block-DP over architectural cliques, sparse pairwise QUBO, top-K Hessian covering

---

## Quick Start

```bash
export MODEL_PATH=/path/to/Qwen3.6-35B-A3B
export WORK_DIR=./dq-runs/qwen36
export FORMATS=NVFP4,MXFP8_E4M3,BF16
export TARGET_BITS=4.75

./prismaquant/run-pipeline.sh
```

Then serve with standard vLLM.

---

## Relevance to DGX Spark / Community Recipe Registry

PrismaQuant is directly relevant to the community-recipe-registry because:

- It produces **NVFP4 checkpoints** — the native quantization format for Blackwell GPUs (DGX Spark's GB10)
- Its mixed-precision allocations are typically **2–11% smaller** than uniform NVFP4 at **better quality**
- It supports streaming quantization of models up to **671B params** (DeepSeek-V4-Flash)
- Its output is standard `compressed-tensors` that vLLM serves — the same serving stack used in these recipes

---

## References

- Paper draft: `paper/main.pdf` in the repo
- HuggingFace artifact: [`rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
- License: MIT

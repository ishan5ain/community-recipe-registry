# qwen3.6-27b-nvfp4-vllm-ishan5ain

**Qwen3.6-27B-NVFP4** (NVIDIA 4-bit floating-point) on a single DGX Spark GB10
(128 GB unified memory, Blackwell SM 12.1).

NVFP4 is a 4-bit floating-point format native to Blackwell GPUs. Unlike INT4
(integer), NVFP4 retains floating-point semantics with a shared exponent and
compact mantissa, giving higher dynamic range and more stable convergence.
The Blackwell TensorCores (`tcgen05.mma` instructions) natively support FP4
matrix operations — no dequantization overhead during matmul.

## Quick Start

```bash
sparkrun run @community/qwen3.6-27b-nvfp4-vllm-ishan5ain --solo
```

Or from the local file:

```bash
sparkrun run recipes/qwen3.6-27b/ishan5ain/qwen3.6-27b-nvfp4-vllm-ishan5ain.yaml --solo
```

## VRAM Estimation

| Component | Size |
|-----------|------|
| Model weights (NVFP4) | ~13.5 GB |
| KV cache (FP8, 32K ctx) | ~4 GB |
| Activation memory + overhead | ~4 GB |
| **Total** | **~21.5 GB** |
| Available (80% of 128 GB) | ~96.8 GB |
| Headroom | ~75 GB |

NVFP4's ~13.5 GB weight footprint is roughly **half** of the FP8 version
(~27 GB). This frees substantial memory for longer context lengths, higher
concurrency, or speculative decoding.

## Customisation

### Increasing context length

The recipe starts with `max_model_len: 32768` as recommended by Unsloth.
You can safely increase it:

```bash
sparkrun run @community/qwen3.6-27b-nvfp4-vllm-ishan5ain \
  --solo \
  -o max_model_len=131072
```

At 131K context with `gpu_memory_utilization: 0.8`, expect:
- KV cache: ~16 GB
- Total: ~29.5 GB
- Plenty of headroom remains

### Adding MTP speculative decoding

Qwen3.6 ships with built-in MTP (Multi-Token Prediction) heads. To enable:

1. Add to defaults:
   ```yaml
   speculative_config: '{"method":"mtp","num_speculative_tokens":2}'
   ```
2. Add to command:
   ```yaml
   --speculative-config '{speculative_config}'
   ```

Then override at runtime:
```bash
sparkrun run @community/qwen3.6-27b-nvfp4-vllm-ishan5ain \
  --solo \
  -o speculative_config='{"method":"mtp","num_speculative_tokens":2}'
```

> ⚠️ MTP on NVFP4 hasn't been benchmarked on GB10 yet. The MTP heads share
> weights with the main model, so their overhead is minimal (~500 MB). However,
> NVFP4 + MTP may behave differently than FP8 + MTP due to the lower-precision
> weights. If acceptance rate is below 50%, turn MTP off.

### Increasing GPU memory for KV cache

```bash
sparkrun run @community/qwen3.6-27b-nvfp4-vllm-ishan5ain \
  --solo \
  -o gpu_memory_utilization=0.85
```

Note: On GB10's unified memory, `gpu_memory_utilization` controls the fraction
of all 128 GB that vLLM reserves. Higher values = more KV cache = longer
context, but less room for other processes.

## Known Issues & Pitfalls

### 1. Load format is auto-detected

NVFP4 is detected automatically by vLLM >= 0.19.0 from the model's
`config.json`. **Do not set `--load-format`** — forcing a format will break
weight loading.

### 2. Dtype must be bfloat16

Activations run in bfloat16 while weights stay in NVFP4. The `--dtype bfloat16`
flag is required. Do not use `--dtype auto` or omit it.

### 3. Performance is unbenchmarked on GB10

NVFP4 on DGX Spark is still maturing in vLLM. The Blackwell FP4 TensorCores
process 2× more elements per cycle than FP8 in theory, but:
- The DeltaNet SSM state is still float32 (not quantized)
- NVFP4 GEMM kernels for SM 12.1 may not be fully optimized yet
- Expect comparable or slightly faster than FP8 at short context, potentially
  better at long context due to reduced memory bandwidth pressure

### 4. Chat template

The model's tokenizer has the Qwen3.6 chat template built in. If you encounter
template errors, override with:
```
--chat-template unsloth.jinja
```

## Author

[@ishan5ain](https://github.com/ishan5ain)

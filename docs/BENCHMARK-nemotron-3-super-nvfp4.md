# Benchmark Report: NVIDIA Nemotron-3-Super-120B-A12B-NVFP4

**Date:** 2026-05-26  
**Recipe:** `@eugr/nemotron-3-super-nvfp4`  
**Server:** `http://127.0.0.1:8000`  
**Framework:** llama-benchy v0.3.7  
**Precision:** NVFP4 weights + FP8 KV cache  
**Prefix Caching:** Enabled  
**GPU:** NVIDIA DGX Spark (GB10, 96 GiB unified memory)

---

## 1. Workload Identity

| Field | Detail |
|---|---|
| Model | NVIDIA Nemotron-3-Super-120B-A12B-NVFP4 |
| Backend | vLLM with FlashInfer |
| Tensor Parallelism | TP=1 |
| GPU Memory (model) | 69.59 GiB |
| KV Cache Memory | 25.12 GiB available, 1,079,978 tokens |
| Max Model Len | 262,144 tokens |
| Max Concurrency (KV) | 4.12× for 262K requests |

---

## 2. Cold Start Timeline

| Phase | Duration |
|---|---|
| Weight loading | 70.9 s |
| Model load (total) | 75.6 s |
| `torch.compile` | 43.6 s |
| Profiling / warmup | 67.1 s |
| FlashInfer autotuning (fp8_gemm + fp4_gemm) | ~260 s |
| CUDA graph capture | 8 s |
| **Total → server ready** | **~7.5 min** |

---

## 3. Prompt Processing (Prefill)

| Depth | Concurrency | Phase | PP Throughput | Notes |
|:----:|:-----------:|:-----:|:-------------:|:------|
| 128 | 1 | Cold (first-time) | **155.8 tok/s** | Full 2048-token prefill, no cache |
| 128 | 1 | Cached | **765.0 tok/s** | Prefix cache hit → ~5× speedup |
| 128 | 2 | Cold | **155.3 tok/s** | Slightly lower under load |
| 128 | 2 | Cached | **678.9 tok/s** | Contention reduces cached throughput |
| 1024 | 1 | Warm | **732.8 tok/s** | System warmed from prior runs |
| 1024 | 1 | Cached | **566.3 tok/s** | Longer depth = more to recompute |

---

## 4. Token Generation (Decode)

| Concurrency | Output Length | Aggregate TG | Per-Request TG | Peak tok/s |
|:-----------:|:-------------:|:------------:|:--------------:|:----------:|
| 1 | 128 tok | **9.47 tok/s** | 9.47 tok/s | 10 |
| 2 | 128 tok | **12.88 tok/s** | ~7.20 tok/s | 18 |
| 1 | 1024 tok | **9.47 tok/s** | 9.47 tok/s | 10 |

Decode speed is **very stable** — ~9.5 tok/s per request regardless of output length. Under concurrency=2, aggregate throughput rises to ~12.9 tok/s, but per-request throughput drops to ~7.2 tok/s due to batching contention.

---

## 5. Time-to-First-Token (TTFT)

| Depth | Concurrency | TTFT (mean) | Range |
|:----:|:-----------:|:-----------:|:------|
| 128 | 1 | **988 ms** | 595–1,381 ms |
| 128 | 2 | **1,192 ms** | 580–2,324 ms ⚠️ |
| 1024 | 1 | **1,403 ms** | 1,392–1,413 ms |

### End-to-End Latency (incl. generation)

| Depth | Concurrency | E2E Time | Breakdown |
|:----:|:-----------:|:--------:|:----------|
| 128 | 1 | 2,853 ms | 988 ms prefill + ~1,865 ms gen |
| 128 | 2 | 4,706 ms | 1,192 ms prefill + ~3,514 ms gen (queued) |
| 1024 | 1 | 3,622 ms | 1,403 ms prefill + ~2,219 ms gen |

---

## 6. Performance Observations

### Strengths
- **Consistent decode**: 9.47 tok/s with ±0.01 std dev at single concurrency
- **Prefix caching effective**: 3.5–5× speedup on re-prefill
- **Reasonable cold TTFT**: ~1 s for 2048-token prompts

### Bottlenecks
1. **Cold prefill is the bottleneck**: 156 tok/s for 2048 tokens accounts for ~13 s in total response cycle
2. **Sub-linear scaling under concurrency**: c=1 → 9.5, c=2 → 12.9 tok/s (+36% only)
3. **Small-batch perf cliff**: Log warning about missing tuned config for shapes like `[1, 10, 1024]` — falls back to unoptimized `CutlassFp8GemmRunner tactic=-1`
4. **TTFT variance under load**: 2.3× spread (580–2,324 ms) at concurrency=2

### Known Issues (from server logs)
- **Uncalibrated FP8 KV cache** — q_scale=1.0 and prob_scale=1.0 may affect output quality
- **Missing Mamba SSU tuned config** for GB10 — mamba layers run unoptimized defaults (`headdim=64, dstate=128, device_name=NVIDIA_GB10`)
- **`shm_broadcast` warnings** during startup — non-critical synchronization delays

---

## 7. Recommendations

| # | Recommendation | Expected Benefit |
|---|---------------|-----------------|
| 1 | Run `sparkrun tune` with expanded `max_num_tokens` / `tuning_buckets` | Eliminate small-batch perf cliff |
| 2 | Calibrate FP8 KV cache scaling factors | Improve output quality |
| 3 | Tune Mamba SSU for GB10 | Better Mamba layer performance |
| 4 | Tweak `max_num_seqs` (currently 10) | May improve batching efficiency |
| 5 | Warm up with representative prompts before serving | Leverage prefix caching from request 1 |

---

## 8. Methodology

- **Profile**: `/tmp/light-benchmark.yaml` (custom, light suite)
- **Tested configs**:
  - Prompt: 2048 tokens (pp)
  - Output lengths: 128, 1024 tokens (depth)
  - Concurrency: 1, 2
  - Runs per config: 2
  - Prefix caching: enabled
- **Skipped**: concurrency=5, 10 and depth=32768+ (reserved for full benchmark)

---

*Generated from `sparkrun benchmark run --skip-run --no-stop --profile /tmp/light-benchmark.yaml @eugr/nemotron-3-super-nvfp4`*

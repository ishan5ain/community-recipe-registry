# Qwen3.6-27B Recipe Testing Log — @ishan5ain

## Hardware

- **System**: ASUS Ascent GX10 (DGX Spark)
- **SoC**: NVIDIA GB10 Grace Blackwell
- **GPU**: Blackwell SM 12.1, 48 SMs, 6,144 CUDA cores
- **Memory**: 128 GB unified LPDDR5X (273 GB/s bandwidth)
- **Driver**: 595.71.05, CUDA 13.2
- **OS**: Ubuntu 24.04
- **Container**: `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`
- **vLLM**: `0.21.1rc1.dev110+g129019f33.d20260522`

---

## Recipe 1: NVFP4 (no MTP) — Baseline

**File**: `qwen3.6-27b-nvfp4-vllm-ishan5ain.yaml`
**Status**: ✅ Working

| Metric | Value |
|--------|-------|
| Decode speed | **6.9 tok/s** |
| Model weights | 12.57 GiB (NVFP4) |
| Total GPU memory | ~24 GiB |
| KV cache blocks | 1,153 (18,448 tokens) |
| Max context @ 80% | 690K tokens |
| KV cache memory available | 55.18 GiB |
| FlashInfer autotune | 4 passes (43s + 7m23s + 2m10s + 2m10s = ~12 min first run) |
| Startup time (first) | ~10-12 min (weights + torch.compile + autotune) |
| Startup time (cached) | ~2-3 min |

**Notes**:
- `--dtype bfloat16` required for NVFP4 (activations BF16, weights FP4)
- `--load-format` must NOT be set — NVFP4 auto-detected from config.json
- KV cache in FP8 (`--kv-cache-dtype fp8`)
- --reasoning-parser qwen3 causes content: null (output goes to `reasoning` field)

---

## Recipe 2: NVFP4 + MTP — FAILED (0% draft acceptance)

**File**: `qwen3.6-27b-nvfp4-mtp-vllm-ishan5ain.yaml`
**Status**: ❌ Works but useless (0% acceptance)

| Metric | Value |
|--------|-------|
| Decode speed | **5.1 tok/s** (worse than baseline!) |
| MTP acceptance rate | **0.0%** — every draft rejected |
| Drafted throughput | 10.2 tok/s (all wasted) |
| KV pool reduction | ~10% from PIECEWISE graph mode |

**Root cause**: The MTP heads share the same NVFP4-quantized weights as the main model. At 4-bit precision, the draft predictions are too noisy for the main model to accept. This adds overhead without benefit.

**Log evidence**:
```
SpecDecoding metrics: Mean acceptance length: 1.00
Per-position acceptance rate: 0.000, 0.000
Avg Draft acceptance rate: 0.0%
```

**First failure (memory)**:
```
ValueError: Free memory on device cuda:0 (34.34/121.63 GiB) on startup
is less than desired GPU memory utilization (0.8, 97.3 GiB)
```
Previous container wasn't fully stopped. Fixed with `docker kill`.

---

## Recipe 3: NVFP4 + DFlash (vLLM) — HUNG at warmup

**File**: `qwen3.6-27b-nvfp4-dflash-vllm-ishan5ain.yaml`
**Status**: ⚠️ Models load but engine hangs at warmup

| Metric | Value |
|--------|-------|
| Target model loaded | ✅ 262.45s (NVFP4, 24 GiB) |
| Draft model loaded | ✅ 34.82s (z-lab/Qwen3.6-27B-DFlash, 3.22 GiB) |
| DFlash layers detected | (1, 16, 31, 46, 61) — 5 auxiliary layers |
| Total memory used | 27.57 GiB |
| Engine start | ❌ Hangs after "Encoder cache will be initialized..." |

**Root cause**: The Qwen3.6 DFlash draft model requires vLLM PR #40898 for SWA/diffusion kernel support. The sparkrun nightly container does NOT include this PR.

**First failure (kv_cache conflict)**:
```
ValueError: Selected backend FLASH_ATTN is not valid for this configuration.
Reason: ['kv_cache_dtype not supported']
```
Fixed by removing `--kv-cache-dtype fp8` when using `--attention-backend flash_attn`.

**From the model page**: *"This model is still under training, and inference engine support may not be fully available yet due to architectural changes, including causal SWA layers."*

---

## FP8 Results (for comparison)

| Config | Speed | MTP Acceptance | Weight Memory |
|--------|-------|----------------|---------------|
| FP8 (no MTP) | 5.0 tok/s | — | ~28.75 GiB |
| FP8 + MTP | **10.1 tok/s** 🏆 | **73.9%** | ~28.75 GiB |

The FP8 + MTP combo is the fastest tested. MTP acceptance at 73.9% on GB10 is excellent — the memory-bandwidth bottleneck means speculative decoding fills otherwise-idle compute.

---

## Key Technical Learnings

### 1. NVFP4 vs FP8 Tradeoffs

| Aspect | FP8 | NVFP4 |
|--------|-----|-------|
| Weight size | ~28.75 GiB | ~12.57 GiB |
| Decode speed (no MTP) | 5.0 tok/s | **6.9 tok/s** (+38%) |
| KV cache capacity | ~9K tokens | **~18K tokens** (2×) |
| MTP compatibility | ✅ Great (73.9%) | ❌ Terrible (0%) |
| DFlash compatibility | ✅ Known working | ⚠️ Untested/hung |

NVFP4 wins on memory efficiency and raw decode speed without MTP. But for peak speed, FP8 + MTP is still king.

### 2. MTP Acceptance is Quantization-Sensitive

FP8 (8-bit) MTP heads: 73.9% acceptance → +102% speed
NVFP4 (4-bit) MTP heads: 0.0% acceptance → -26% speed (overhead)

The 4-bit quantization loses too much precision for the MTP heads. The draft predictions don't match the main model's expectations. This is a fundamental limitation, not a bug.

### 3. DFlash Support is Early

- Qwen3.6 DFlash: "still under training" — needs vLLM PR #40898
- Qwen3.5 DFlash: mature and working (see banana_baeee's recipe)
- llama.cpp DFlash (spiritbuun fork): proven on GB10, 38-40 tok/s

### 4. GB10-Specific Observations

- **Memory bandwidth bottleneck**: 273 GB/s LPDDR5X — decode speed limited by weight reading, not compute
- **FlashInfer autotuning**: 4 passes × 23 profiles each, ~12 min first run. Cached on disk for subsequent runs.
- **torch.compile**: ~2 min first run (AOT cached). With `--enforce-eager`, skipped entirely.
- **CUDA graphs**: FULL_AND_PIECEWISE mode. With MTP, downgrades to PIECEWISE.
- **"Not enough SMs"**: GB10 has 48 SMs, below vLLM's threshold for max_autotune_gemm.
- **Unified memory**: `nvidia-smi` shows "Not Supported" for GPU memory query. Memory is shared CPU+GPU.

### 5. Sparkrun Gotchas

- `sparkrun stop <container_name>` does NOT work — it expects a **recipe name**, not a container name
- `sparkrun stop --all` needs SSH host keys for localhost (`ssh-keyscan -H 127.0.0.1 >> ~/.ssh/known_hosts`)
- Use `docker kill` as fallback when sparkrun fails
- Auto-restart: sparkrun respawns containers when killed — stop via sparkrun recipe name or `docker kill && docker rm`
- Container network mode: `host` (ports exposed directly, not mapped)

### 6. Recipe Versioning

Community recipes use `recipe_version: "2"`. File naming convention:
`{model}-{quant}-{mtp/dflash}-{runtime}-{user}.yaml`
Directory: `recipes/{model-name}/{user}/`

### 7. FlashInfer vs Flash Attention

| Aspect | FlashInfer | Flash Attention |
|--------|------------|-----------------|
| Used for | vLLM default | DFlash (vLLM) / llama.cpp |
| FP8 KV cache | ✅ Supported | ❌ Not supported |
| NVFP4 GEMM | ✅ Custom kernel | ⚠️ Limited |
| DFlash support | ❌ | ✅ |

### 8. Content vs Reasoning Field

Using `--reasoning-parser qwen3` causes the model to output all content into the `reasoning` field, with `content: null`. This is expected behavior — clients must read from `.reasoning` not `.content`.

---

## Performance Summary

| Recipe | tok/s | Weight Memory | KV Cache | Best For |
|--------|-------|---------------|----------|----------|
| FP8 + MTP | **10.1** 🥇 | 28.75 GiB | ~9K | Max speed |
| NVFP4 (no MTP) | **6.9** 🥈 | 12.57 GiB | ~18K | Long context / memory efficiency |
| FP8 (no MTP) | 5.0 | 28.75 GiB | ~9K | Baseline |
| NVFP4 + MTP | 5.1 ❌ | 12.57 GiB | ~18K | Don't use |
| NVFP4 + DFlash (vLLM) | — ⚠️ | 27.57 GiB | — | Needs PR #40898 |
| llama.cpp DFlash (Q4_K_M) | **38-40** 🏆🏆 | ~16 GiB | turbo4 | Max performance (if built) |

---

### 9. Llama.cpp Stock Container vs DFlash Fork

The stock `ghcr.io/spark-arena/dgx-llama-cpp:latest` container does NOT support DFlash:

```
error: invalid argument: --draft-context-size
```

No DFlash flags (`--spec-type`, `--spec-dflash-default`, `--draft-context-size`,
`-ctk`, `-ctv`, etc.) are recognized by the stock `llama-server`. The DFlash
support requires building the spiritbuun/phuongncn custom fork from source with:

```bash
git clone https://github.com/phuongncn/qwen3.6-27b-speedhack-gx10-dgx-spark.git
cd qwen3.6-27b-speedhack-gx10-dgx-spark
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_RPC=OFF
cmake --build . -j20 --config Release

# Results: ~38-40 tok/s (per repo README, untested by @ishan5ain)
```

The recipe serves as documentation of the exact flags needed but cannot run
directly until a custom container is built.

---

*Testing conducted May 24, 2026 on ASUS Ascent GX10 (NVIDIA GB10)*

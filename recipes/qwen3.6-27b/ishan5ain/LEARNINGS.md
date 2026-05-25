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

**Root cause**: The `unsloth/Qwen3.6-27B-NVFP4` model uses the `compressed-tensors` quantization format, which **drops the MTP head during export** — the MTP weights are simply absent. The 0% acceptance is not a quantization precision issue; the draft head outputs garbage because its weights are missing.

**Important**: There is a separate quantization format called `modelopt` (NVIDIA's ModelOpt path) that **preserves the MTP head in bf16**. Models using `modelopt` NVFP4, such as `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (text-only, ~80 tok/s claimed on RTX 5090) or `sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP` (with vision), can have fully working MTP speculative decoding.

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

## Recipe 4: llama.cpp DFlash — Baseline (no thinking)

**File**: `qwen3.6-27b-q4km-dflash-llamacpp-ishan5ain.yaml`
**Runtime**: llama-cpp (phuongncn speedhack fork)
**Container**: `dgx-llama-cpp-dflash:latest` (custom build)
**Status**: ✅ Working

| Metric | Value |
|--------|-------|
| SM arch build fix | Initial build `120` → rebuilt with `121a` (blackwell SM 12.1) |
| Containers built | 2: binary-copy + rebuilt for SM 12.1a |
| HTTPS support | ❌ Not compiled in (no libssl-dev); uses `-md` + curl for draft model |
| Draft model download | ✅ curl via HuggingFace (1.7 GB Q8_0) |
| Target model | unsloth/Qwen3.6-27B-GGUF:Q4_K_M (16 GB, pre-downloaded by sparkrun) |
| GPU memory | ~15.3 GiB target + ~1.6 GiB draft + ~4.1 GiB KV cache (turbo4) + ~1.2 GiB recurrent = ~22 GiB total |
| DFlash token acceptance | **58.3%** (matches speedhack author's 52-61% range for Q4_K_M) |
| DFlash call acceptance | **75.3%** |
| Avg spec cycle | ~250ms (draft ~33ms, verify ~190ms) |

### Benchmark Results

| Scenario | tok/s | Tokens | Time |
|----------|:-----:|:------:|:----:|
| HTML/JS coding (400 tok) | **17.3** | 400 | 23.1s |
| Python coding (500 tok) | **25.7** | 500 | 19.4s |
| Short chat (~150 tok) | 1.6* | 4 | 2.4s |
| Medium context (300 tok) | **11.2** | 300 | 26.6s |
| Sustained 2048 tok | **10.1** | 2048 | 202.2s |

*\*Short chat only generated 4 tokens (just the word "Paris") — too short to measure meaningfully.*

**Key finding**: Python coding at 25.7 tok/s closely matches the speedhack author's claimed 24-25 tok/s. HTML/JS and sustained are below the claimed 38-40 and 27-29 tok/s. Gap may be due to content type, prompt design, or batch-size capping.

---

## Recipe 5: llama.cpp DFlash — Thinking variant

**File**: `qwen3.6-27b-q4km-dflash-thinking-llamacpp-ishan5ain.yaml`
**Runtime**: llama-cpp (phuongncn speedhack fork)
**Container**: `dgx-llama-cpp-dflash:latest` (custom build)
**Status**: ✅ Working

Changes from baseline:
- `--reasoning on` + `--reasoning-format deepseek` — enables Qwen3.6 thinking mode
- `--chat-template-kwargs '{"preserve_thinking":true}'` — retain reasoning context across turns
- `--temp 0.6 --top-k 20 --top-p 0.95 --min-p 0.0` — official Qwen3.6 precise coding params
- No `--no-mmap` — matches speedhack author's setup

| Metric | Value |
|--------|-------|
| DFlash token acceptance | **56.0%** |
| DFlash call acceptance | **73.6%** |
| Avg spec cycle | ~251ms (similar to baseline) |
| Reasoning field | ✅ `reasoning_content` populated via `deepseek` format |

### Benchmark Results

| Scenario | tok/s | Tokens | Time |
|----------|:-----:|:------:|:----:|
| HTML/JS coding (400 tok) | **17.0** | 400 | 23.5s |
| Python coding (500 tok) | **21.9** | 500 | 22.7s |
| Short chat (150 tok) | **12.8** | 106 | 8.3s |
| Medium context (300 tok) | **12.4** | 300 | 24.0s |
| Sustained 2048 tok | **11.0** | 2048 | 186.0s |

**Thinking overhead**: ~1-3 tok/s slower than baseline due to thinking token generation. Output quality improves significantly — short chat now generates 106 tokens (with thinking) vs 4 in baseline. The `reasoning_content` field contains the model's chain-of-thought.

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

| Aspect | FP8 | NVFP4 (compressed-tensors) | NVFP4 (modelopt) |
|--------|-----|--------------------------|-------------------|
| Weight size | ~28.75 GiB | ~12.57 GiB | ~18.65 GiB |
| Decode speed (no MTP) | 5.0 tok/s | **6.9 tok/s** (+38%) | — |
| Decode speed (MTP) | 10.1 tok/s | 5.1 tok/s (0% accept) | **15-17 tok/s** ✅ |
| KV cache capacity | ~9K tokens | **~18K tokens** (2×) | **~1.1M tokens** (59GiB) |
| MTP compatibility | ✅ 73.9% | ❌ 0% (head stripped) | ✅ **64-91%** 🎉 |
| DFlash compatibility | ✅ Known working | ⚠️ Untested/hung | — |

**Updated finding**: NVFP4+MTP with the modelopt format is now the fastest vLLM-based option on GB10 (15-17 tok/s), beating FP8+MTP (10.1 tok/s) by **50-68%**.

### 2. MTP Acceptance Depends on Quantization Format (Not Just Bits)

Our NVFP4+MTP failure (0% acceptance) was caused by the **quantization format**, not the bit depth:

| Quant format | MTP head preservation | MTP works? | Example model |
|---|---|---|---|
| `compressed-tensors` | ❌ **Dropped during export** | ❌ 0% acceptance | `unsloth/Qwen3.6-27B-NVFP4` |
| `modelopt` (ModelOpt) | ✅ **Restored in bf16** | ✅ **64-91% on GB10** 🎉 | `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` |
| Unquantized (bf16) | ✅ Intact | ✅ 73.9% (FP8+MTP) | `Qwen/Qwen3.6-27B` (FP8 quant) |

**Root cause**: The `compressed-tensors` export path strips all non-essential weights including the MTP heads. The `modelopt` export path preserves the MTP head in bf16 while keeping the main weights in NVFP4. This is a toolchain issue, not a fundamental precision limitation.

### 3. Quantization Format: `modelopt` vs `compressed-tensors`

For NVFP4 quantization on Blackwell (GB10), there are two competing formats:

| Aspect | `modelopt` (NVIDIA ModelOpt) | `compressed-tensors` |
|---|---|---|
| MTP head | ✅ Preserved in bf16 | ❌ Stripped |
| Vision tower | ✅ Preserved | ✅ Preserved (in VLM models) |
| vLLM backend | SM120 native path | compressed-tensors loader |
| Known good at | ✅ GB10 (sm_121a) — tested working via `FlashInferCutlassNvFp4LinearKernel` | GB10 (sm_121a) tested |
| Startup speed | Faster (native path) | Slower (detour) |
| MTP on GB10 | ✅ Working — **64-91% acceptance rate** 🎉 | ❌ 0% acceptance (MTP head stripped) |
| Models | `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`, `sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP` | `unsloth/Qwen3.6-27B-NVFP4` |

**The `modelopt` format FIXES NVFP4+MTP on GB10** ✅ — The SM120 native kernel path works via `FlashInferCutlassNvFp4LinearKernel` on SM 121a. The `VLLM_TEST_FORCE_FP8_MARLIN=1` env var was set but not used (CUTLASS path was selected).

### 4. `VLLM_TEST_FORCE_FP8_MARLIN` Is Not Needed on Current vLLM

The `VLLM_TEST_FORCE_FP8_MARLIN=1` env var was carried over from Recipe 1 (baseline) and earlier community guidance for NVFP4 on SM 121a. With the current vLLM nightly (`0.21.1rc1`), the NVFP4 backend automatically selects `FlashInferCutlassNvFp4LinearKernel` without it. The thinking variant confirmed this by running cleanly without the env var.

**Recommendation**: Omit `VLLM_TEST_FORCE_FP8_MARLIN` in new recipes. Only add if specific CUDA errors occur with the CUTLASS path.

### 5. Thinking Mode Overhead Varies by Task Type

Enabling thinking mode (`--reasoning-parser qwen3`) adds overhead that depends heavily on the task:

| Task type | Throughput impact | Reason |
|---|---|---|
| Creative/generative (HTML/JS) | **-48%** (16.9→8.8 tok/s) | Model spends many tokens reasoning about design choices before generating |
| Coding (Python, algorithms) | **~-6%** (16.1→15.1 tok/s) | Minimal reasoning needed for straightforward code tasks |
| Factual Q&A | **~same** | Short reasoning, quick answer |
| Sustained generation | **~same** | Once context is established, thinking overhead is amortized |

**Tradeoff**: Thinking mode enables `reasoning_content` in responses and improves output quality for complex tasks, but at a throughput cost that varies significantly by content type.

### 6. `--language-model-only` Is Required for Text-Only VLM-Derived Models

Models like `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` have a `config.json` that reports `image-text-to-text` even though the vision tower is stripped. Without `--language-model-only`, vLLM tries to load a multimodal processor and crashes with:

```
OSError: Can't load image processor for ... missing preprocessor_config.json
```

**Fix**: Always include `--language-model-only` when serving models that are based on a VLM architecture but are text-only.

### 7. DFlash Support is Early

- Qwen3.6 DFlash: "still under training" — needs vLLM PR #40898
- Qwen3.5 DFlash: mature and working (see banana_baeee's recipe)
- llama.cpp DFlash (phuongncn speedhack fork): proven on GB10, 38-40 tok/s (claimed)
- llama.cpp DFlash tested: Python coding matches claim (25.7 tok/s); HTML/JS and sustained below (see Recipe 4/5)

### 8. GB10-Specific Observations

- **Memory bandwidth bottleneck**: 273 GB/s LPDDR5X — decode speed limited by weight reading, not compute
- **FlashInfer autotuning**: 4 passes × 23 profiles each, ~12 min first run. Cached on disk for subsequent runs.
- **torch.compile**: ~2 min first run (AOT cached). With `--enforce-eager`, skipped entirely.
- **CUDA graphs**: FULL_AND_PIECEWISE mode. With MTP, downgrades to PIECEWISE.
- **"Not enough SMs"**: GB10 has 48 SMs, below vLLM's threshold for max_autotune_gemm.
- **Unified memory**: `nvidia-smi` shows "Not Supported" for GPU memory query. Memory is shared CPU+GPU.

### 9. Sparkrun Gotchas

- `sparkrun stop <container_name>` does NOT work — it expects a **recipe name**, not a container name
- `sparkrun stop --all` needs SSH host keys for localhost (`ssh-keyscan -H 127.0.0.1 >> ~/.ssh/known_hosts`)
- Use `docker kill` as fallback when sparkrun fails
- Auto-restart: sparkrun respawns containers when killed — stop via sparkrun recipe name or `docker kill && docker rm`
- Container network mode: `host` (ports exposed directly, not mapped)

### 10. Recipe Versioning

Community recipes use `recipe_version: "2"`. File naming convention:
`{model}-{quant}-{mtp/dflash}-{runtime}-{user}.yaml`
Directory: `recipes/{model-name}/{user}/`

### 11. FlashInfer vs Flash Attention

| Aspect | FlashInfer | Flash Attention |
|--------|------------|-----------------|
| Used for | vLLM default | DFlash (vLLM) / llama.cpp |
| FP8 KV cache | ✅ Supported | ❌ Not supported |
| NVFP4 GEMM | ✅ Custom kernel | ⚠️ Limited |
| DFlash support | ❌ | ✅ |

### 12. Content vs Reasoning Field

Using `--reasoning-parser qwen3` causes the model to output all content into the `reasoning` field, with `content: null`. This is expected behavior — clients must read from `.reasoning` not `.content`.

---

## Recipe 6: NVFP4 + MTP (modelopt format) — WORKING 🎉

**File**: `qwen3.6-27b-nvfp4-mtp-modelopt-vllm-ishan5ain.yaml`
**Model**: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
**Runtime**: vllm
**Container**: `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`
**Status**: ✅ Working — MTP acceptance rate **64-91%** (vs 0% in Recipe 2)

**Key differences from Recipe 2** (which got 0% MTP acceptance):
- Uses `modelopt` quantization (not `compressed-tensors`)
- **MTP head preserved in bf16** (not stripped) — this is why it works
- Text-only (vision tower stripped) — smaller weight footprint
- Claimed ~80 tok/s on RTX 5090 with MTP (we get ~15-17 tok/s on GB10)

| Metric | Value |
|--------|-------|
| Decode speed | **15-17 tok/s** (varies by scenario) |
| Model weights | 18.65 GiB (NVFP4 + MTP head in bf16) |
| KV cache | 1,141,915 tokens (59.68 GiB available) |
| NVFP4 GEMM backend | `FlashInferCutlassNvFp4LinearKernel` ✅ |
| MTP acceptance rate | **64-91%** across all requests |
| Per-position acceptance | Pos 1: 75-94%, Pos 2: 53-89% |
| MTP draft tokens | 2 (num_speculative_tokens) |
| Spec decode throughput | ~7-10 tok/s accepted, ~11-12 tok/s drafted |
| CUDA graph mode | PIECEWISE (FULL not supported with spec-decode + FlashInfer) |
| Startup time (first) | ~16 min (230s model load + 69s torch.compile + ~12 min FlashInfer autotune) |
| Startup time (cached) | ~4-5 min (same as Recipe 1) |

**First run failure**: Missing `preprocessor_config.json` — model's `config.json` reports `image-text-to-text` even though vision tower is stripped.
- **Fix**: Added `--language-model-only` flag

### Benchmark Results

| Scenario | tok/s | Tokens | Time |
|----------|:-----:|:------:|:----:|
| HTML/JS coding (400 tok) | **16.9** | 400 | 23.7s |
| Python coding (500 tok) | **16.1** | 500 | 30.9s |
| Short chat (150 tok) | **15.4** | 117 | 7.6s |
| Sustained 500 tok | **15.0** | 500 | 33.2s |

### MTP Acceptance Over Time

```
1st request (fresh):  94.4%, 88.9% → 91.7% avg
2nd request:          90.0%, 83.3% → 86.7% avg
3rd request:          91.2%, 86.0% → 88.6% avg
4th request (sustained): 86.4%, 67.8% → 77.1% avg
5th request (sustained): 75.0%, 53.3% → 64.2% avg
```

**Observations**:
- First-token acceptance is excellent (~86-94%) — MTP confidently predicts the next token
- Second-token acceptance drops at sustained context (~53-67%) — MTP with only 2 tokens has limited lookahead
- Acceptance decreases as context grows — the draft predictions become less accurate
- Overall throughput is **2.2-2.5× faster** than Recipe 1 (NVFP4 no MTP: 6.9 tok/s)

**Important**: The `num_speculative_tokens=2` means MTP runs 2 forward passes per decode step. The vLLM warning says: "Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate." This is visible in the second-position dropoff.

**Risks on DGX Spark**:
- The `modelopt` NVFP4 format uses SM120 native kernel path — works on SM 121a via `FlashInferCutlassNvFp4LinearKernel`
- `VLLM_TEST_FORCE_FP8_MARLIN=1` was set but the log shows `CUTLASS` (not Marlin) being used — the env var may not be needed for this model

### Thinking Variant of Recipe 6

**File**: `qwen3.6-27b-nvfp4-mtp-thinking-vllm-ishan5ain.yaml`
**Status**: ✅ Working — Qwen3.6 official sampling params with preserve_thinking

Changes from Recipe 6:
- `--served-model-name qwen3.6-nvfp4-mtp-thinking`
- Client defaults: temp=0.6, top_p=0.95, top_k=20, min_p=0.0 (precise coding)
- `chat_template_kwargs`: `{"preserve_thinking": true}`
- Removed `VLLM_TEST_FORCE_FP8_MARLIN` env var (confirmed unused)

| Scenario | tok/s | Tokens | Time |
|----------|:-----:|:------:|:----:|
| HTML/JS coding (400 tok) | **8.8** | 400 | 45.0s |
| Python coding (500 tok) | **15.1** | 500 | 32.9s |
| Short chat (150 tok) | **15.3** | 87 | 5.7s |
| Sustained 500 tok | **15.0** | 500 | 33.3s |

**MTP stats**: 76-82% acceptance rate (slightly lower than Recipe 6's 64-91%)

**Note**: HTML/JS is significantly slower (8.8 vs 16.9 tok/s) because thinking mode generates extensive reasoning for creative coding tasks, consuming the token budget. Python/sustained are similar to Recipe 6.

---

## Performance Summary

### All Configurations Compared

| Recipe | tok/s (HTML/JS) | tok/s (Python) | tok/s (Sustained) | Weight Memory | Best For |
|--------|:---------------:|:--------------:|:----------------:|:------------:|----------|
| FP8 + MTP | — | — | — | 28.75 GiB | Max raw speed |
| **llama.cpp DFlash (baseline)** | **17.3** | **25.7** 🏆 | **10.1** | ~16 GiB | Python coding |
| **llama.cpp DFlash (thinking)** | **17.0** | **21.9** | **11.0** 🏆 | ~16 GiB | Agentic coding w/ thinking |
| **NVFP4+MTP (modelopt)** | **16.9** | **16.1** | **15.0** | 18.65 GiB | Fast NVFP4 via vLLM |
| **NVFP4+MTP (thinking)** 🆕 | **8.8** | **15.1** | **15.0** | 18.65 GiB | Qwen3.6 official params |
| NVFP4 (no MTP) | 6.9 | — | — | 12.57 GiB | Long context efficiency |
| FP8 (no MTP) | — | — | — | 28.75 GiB | Baseline |
| FP8 + MTP | — | — | — | 28.75 GiB | Max raw speed |
| NVFP4 + MTP (compressed-tensors) | — | — | — | 12.57 GiB | ❌ 0% acceptance |
| NVFP4 + DFlash (vLLM) | — | — | — | 27.57 GiB | ⚠️ Needs PR #40898 |

### llama.cpp DFlash vs Speedhack Claims

| Scenario | Baseline (no thinking) | Thinking variant | Speedhack claim | Match? |
|----------|:---------------------:|:----------------:|:---------------:|:------:|
| HTML/JS coding | 17.3 tok/s | 17.0 tok/s | **38-40 tok/s** | ❌ 2.2× gap |
| Python coding | **25.7 tok/s** | 21.9 tok/s | **24-25 tok/s** | ✅ Baseline matches |
| Short chat | ~1.6 tok/s | **12.8 tok/s** | 23-25 tok/s | ⚠️ Thinking improves |
| Medium context | **11.2 tok/s** | 12.4 tok/s | 20-22 tok/s | ❌ 1.8× gap |
| Sustained 2048 | 10.1 tok/s | **11.0 tok/s** | 27-29 tok/s | ❌ 2.6× gap |

**Key observations**:
- Python coding (25.7 tok/s) matches the speedhack claim — DFlash works correctly
- Sustained and HTML/JS below claims — likely content/prompt/tuning differences
- Thinking mode adds ~1-3 tok/s overhead but enables reasoning for agentic tasks
- Token acceptance rate (56-58%) is within speedhack's reported 52-61% range
- Baseline DFlash Python (25.7 tok/s) is **2.5× faster** than NVFP4 baseline (6.9 tok/s)

---

### 13. Llama.cpp Stock Container vs DFlash Fork

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

## To Investigate

### AEON-7 Qwen3.6-35B-A3B MoE + DFlash (vLLM)

Pre-built vLLM container + NVFP4 DFlash for 35B-A3B MoE on DGX Spark, achieving **116.8 tok/s** single-stream.
Not directly applicable to our 27B dense focus, but worth noting for future exploration.

- Repo: [AEON-7/Qwen3.6-NVFP4-DFlash](https://github.com/AEON-7/Qwen3.6-NVFP4-DFlash)
- Container: `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2`
- Uses custom vLLM patches for SM 121a compatibility
- DFlash not MTP as the speculative backend

---

*Testing conducted May 24-25, 2026 on ASUS Ascent GX10 (NVIDIA GB10)*

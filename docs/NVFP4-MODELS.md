# NVFP4 Models on HuggingFace

> Comprehensive catalog of NVFP4-quantized models available on HuggingFace.
> Generated via `hf models list --search "nvfp4" --sort downloads` (May 2026).
>
> NVFP4 is a 4-bit floating-point format (E2M1 with two-level FP8+FP32 block scaling)
> introduced with NVIDIA Blackwell GPUs. It's the native quantization format for
> DGX Spark (GB10 Blackwell) inference.

## Quick Reference

```bash
# Browse live
hf models list --search "nvfp4" --sort downloads --limit 20 --expand=downloads,likes

# Filter by publisher
hf models list --search "nvfp4" --author nvidia --sort downloads --limit 20
hf models list --search "nvfp4" --author RedHatAI --sort downloads --limit 20
```

---

## 🏆 Top 20 Most Downloaded

| # | Model | Publisher | Downloads | Type |
|---|-------|-----------|-----------|------|
| 1 | **RedHatAI/Qwen3.6-35B-A3B-NVFP4** | RedHatAI | **2.36M** | Text Gen (MoE) |
| 2 | **nvidia/Gemma-4-31B-IT-NVFP4** | NVIDIA | **2.36M** | Text Gen |
| 3 | **nvidia/DeepSeek-R1-0528-NVFP4-v2** | NVIDIA | **1.67M** | Text Gen |
| 4 | **nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4** | NVIDIA | **1.29M** | Any-to-Any (MoE) |
| 5 | **nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4** | NVIDIA | **1.07M** | Text Gen (Latent MoE) |
| 6 | **nvidia/Kimi-K2.5-NVFP4** | NVIDIA | **1.00M** | Text Gen |
| 7 | **nvidia/Gemma-4-26B-A4B-NVFP4** | NVIDIA | **989K** | Text Gen (MoE) |
| 8 | **nvidia/Qwen3.5-397B-A17B-NVFP4** | NVIDIA | **872K** | Text Gen (MoE) |
| 9 | **sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP** | Community | **778K** | Text Gen + MTP |
| 10 | **RedHatAI/gemma-4-26B-A4B-it-NVFP4** | RedHatAI | **723K** | Text Gen (MoE) |
| 11 | **nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4** | NVIDIA | **619K** | Text Gen (MoE) |
| 12 | **AxionML/Qwen3.5-9B-NVFP4** | AxionML | **596K** | Image-Text-to-Text |
| 13 | **lukealonso/MiniMax-M2.7-NVFP4** | Community | **438K** | Text Gen |
| 14 | **unsloth/Qwen3.6-27B-NVFP4** | Unsloth | **365K** | Image-Text-to-Text |
| 15 | **nvidia/Kimi-K2.6-NVFP4** | NVIDIA | **355K** | Text Gen |
| 16 | **LilaRest/gemma-4-31B-it-NVFP4-turbo** | Community | **298K** | Text Gen |
| 17 | **bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4** | Community | **287K** | Text Gen (MoE) |
| 18 | **nvidia/Llama-3.1-8B-Instruct-NVFP4** | NVIDIA | **248K** | Text Gen |
| 19 | **nvidia/Qwen2.5-VL-7B-Instruct-NVFP4** | NVIDIA | **234K** | Vision-Language |
| 20 | **nvidia/MiniMax-M2.7-NVFP4** | NVIDIA | **226K** | Text Gen |

---

## 🏢 NVIDIA Official (nvidia/)

### Large / Frontier Models

| Model | Params (Total) | Active | Downloads | Notes |
|-------|---------------|--------|-----------|-------|
| `nvidia/Qwen3.5-397B-A17B-NVFP4` | 397B | 17B | 872K | Qwen 3.5 MoE |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 120B | 12B | 1.07M | Latent MoE, MTP |
| `nvidia/Qwen3-235B-A22B-Thinking-2507-NVFP4` | 235B | 22B | 22K | Thinking variant |
| `nvidia/DeepSeek-R1-0528-NVFP4-v2` | — | — | 1.67M | v2 adds attn `wo` quant |
| `nvidia/DeepSeek-V3.2-NVFP4` | 671B | 37B | 37K | Latest DeepSeek |
| `nvidia/DeepSeek-V3.1-NVFP4` | 671B | 37B | 16K | |
| `nvidia/DeepSeek-V3-0324-NVFP4` | 671B | 37B | 58K | |
| `nvidia/Kimi-K2.5-NVFP4` | — | — | 1.00M | Moonshot AI |
| `nvidia/Kimi-K2.6-NVFP4` | — | — | 355K | |
| `nvidia/Kimi-K2-Thinking-NVFP4` | — | — | 12K | |
| `nvidia/GLM-5-NVFP4` | — | — | 107K | ZAI GLM-5 |
| `nvidia/Llama-3.1-405B-Instruct-NVFP4` | 405B | 405B | — | Largest official NVFP4 |
| `nvidia/Llama-3.3-70B-Instruct-NVFP4` | 70B | 70B | 63K | Llama 3.3 |

### Mid-Range Models

| Model | Params (Total) | Active | Downloads | Notes |
|-------|---------------|--------|-----------|-------|
| `nvidia/Gemma-4-31B-IT-NVFP4` | 31B | 31B | 2.36M | Multimodal |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | 26B | 4B | 989K | MoE |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | 30B | 3B | 619K | Text gen |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | 30B | 3B | 1.29M | Any-to-any multimodal |
| `nvidia/Qwen3-32B-NVFP4` | 32B | 32B | 60K | Dense |
| `nvidia/Qwen3-30B-A3B-NVFP4` | 30B | 3B | 47K | MoE |
| `nvidia/Qwen3-14B-NVFP4` | 14B | 14B | 34K | Dense |
| `nvidia/Qwen3-8B-NVFP4` | 8B | 8B | 58K | Dense |
| `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | 80B | 3B | 13K | Latest Qwen |
| `nvidia/MiniMax-M2.7-NVFP4` | — | — | 226K | |
| `nvidia/MiniMax-M2.5-NVFP4` | — | — | 139K | |
| `nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4` | 17B | — | 179K | Meta Llama 4 MoE |
| `nvidia/Llama-3.1-8B-Instruct-NVFP4` | 8B | 8B | 248K | Workhorse |

### Vision / Multimodal Models

| Model | Type | Downloads | Notes |
|-------|------|-----------|-------|
| `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` | Vision-Language | 234K | |
| `nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4` | Text-to-Video | — | Wan 14B DiT |
| `nvidia/Phi-4-multimodal-instruct-NVFP4` | Multimodal | — | |

---

## 🔴 RedHatAI

Red Hat is the most active third-party NVFP4 publisher. They often beat NVIDIA to market
with newly released base models.

| Model | Params (Total) | Active | Downloads |
|-------|---------------|--------|-----------|
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | 35B | 3B | **2.36M** |
| `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | 26B | 4B | 723K |
| `RedHatAI/gemma-4-31B-it-NVFP4` | 31B | — | 181K |
| `RedHatAI/Qwen3.5-122B-A10B-NVFP4` | 122B | 10B | 86K |
| `RedHatAI/Qwen3-Next-80B-A3B-Instruct-NVFP4` | 80B | 3B | 54K |
| `RedHatAI/Qwen3-Coder-Next-NVFP4` | — | — | 51K |
| `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` | — | — | 50K |
| `RedHatAI/Qwen3-32B-NVFP4` | 32B | 32B | 46K |
| `RedHatAI/Qwen3-30B-A3B-NVFP4` | 30B | 3B | 27K |
| `RedHatAI/Kimi-K2.6-NVFP4` | — | — | 22K |
| `RedHatAI/Llama-3.1-8B-Instruct-NVFP4` | 8B | 8B | 16K |
| `RedHatAI/Llama-3.3-70B-Instruct-NVFP4` | 70B | 70B | 14K |
| `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4` | 24B | — | 9.1K |
| `RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4` | 235B | 22B | 1.0K |
| `RedHatAI/Qwen3-8B-NVFP4` | 8B | 8B | 2.0K |
| `RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4` | 17B | — | 465 |
| `RedHatAI/Mistral-Large-3-675B-Instruct-2512-NVFP4` | 675B | — | 17 |

---

## 🌐 Notable Community Publishers

| Publisher | Focus / Notable Models | Top Downloads |
|-----------|----------------------|---------------|
| **sakamakismile** | Qwen3.6 + MTP speculative decoding | 778K |
| **AxionML** | Qwen3.5, SGLang-compatible | 596K |
| **lukealonso** | MiniMax-M2.7, GLM-5.1, MiMo-V2.5 | 438K |
| **unsloth** | Unsloth-compressed Qwen3.6 | 365K |
| **LilaRest** | Gemma-4-31B NVFP4-turbo (optimized) | 298K |
| **bg-digitalservices** | Gemma-4 MoE, DGX Spark tagged | 287K |
| **AEON-7** | Uncensored/abliterated Qwen3.6 + Gemma-4 | 207K |
| **Ex0bit** | Qwen3.6 PRISM quantization | 197K |
| **Sehyo** | Qwen3.5 MoE, LLM-Compressor | 196K |
| **GadflyII** | GLM-4.7-Flash MoE | 94K |
| **mmangkad** | Qwen3.6 (ModelOpt) | 67K |
| **GaleneAI** | Llama-3.1 + Aegis content safety | 107K |
| **cosmicproc** | Gemma-4-E4B-it | 38K |
| **enfuse** | Qwen2.5-72B-Instruct | 16K |
| **zdy1995love** | Mistral-Medium-3.5-128B | 13K |

---

## 🎬 Diffusion / Video Models

| Model | Publisher | Type | Notes |
|-------|-----------|------|-------|
| **Lightricks/LTX-2.3-nvfp4** | Lightricks | Text/Image/Video-to-Video | 51K downloads |
| **nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4** | NVIDIA | Text-to-Video | Wan 14B DiT |
| **nvidia/FLUX.1-dev-onnx** | BFL / NVIDIA | Text-to-Image | ONNX format |
| **Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4** | ELM | Few-step Video | Wan-based, FP4 KV-cache |
| **lightx2v/Wan-NVFP4** | lightx2v | Few-step Video | 4-step distilled, real-time on RTX 5090 |

---

## 🧠 DGX Spark Compatibility Notes

### What fits on a single DGX Spark (64GB UMA)

| Model Size | Feasibility | Examples |
|-----------|-------------|----------|
| ≤ 8B param (FP16 → NVFP4) | ✅ Comfortable | Llama-3.1-8B, Qwen3-8B |
| 8B–32B param | ✅ Good | Qwen3-14B, Qwen3-32B, Gemma-4-31B |
| 30B MoE (3B active) | ✅ Excellent | Nemotron-3-Nano, Qwen3-30B-A3B |
| 35B MoE (3B active) | ✅ Good | Qwen3.6-35B-A3B |
| 26B MoE (4B active) | ✅ Good | Gemma-4-26B-A4B |
| 80B MoE (3B active) | ⚠️ Tight, may need `drop_caches` | Qwen3-Next-80B-A3B |
| 120B+ (10B+ active) | ❌ Multi-node needed | Nemotron-3-Super-120B, DeepSeek-V3.1 |

### Tips for DGX Spark

```bash
# If you hit OOM, flush the UMA cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Use --gpu-memory-utilization 0.85-0.88 for most models
# Use tensor-parallel-size 2 for dual-GPU Spark (if available)
```

---

## 📊 Publisher Summary

| Publisher | Models Published | Total Downloads (est.) |
|-----------|-----------------|----------------------|
| **NVIDIA** (nvidia/) | ~25 | ~14M+ |
| **RedHatAI** | ~17 | ~4.2M+ |
| **sakamakismile** | ~5 | ~1.3M+ |
| **lukealonso** | ~7 | ~700K+ |
| **AEON-7** | ~6 | ~500K+ |
| **unsloth** | ~2 | ~495K |
| **bg-digitalservices** | ~3 | ~338K |
| **AxionML** | ~1 | ~596K |
| **LilaRest** | ~1 | ~298K |

---

## 🔍 How to Browse Live

```bash
# List all NVFP4 models sorted by downloads
hf models list --search "nvfp4" --sort downloads --limit 50 --expand=downloads,likes

# List models from a specific author
hf models list --author nvidia --search "nvfp4" --sort downloads --limit 30

# Get model info
hf models info RedHatAI/Qwen3.6-35B-A3B-NVFP4 --expand=downloads,likes,tags

# Search for specific capabilities (vision, video, etc.)
hf models list --search "nvfp4" --filter image-text-to-text --sort downloads --limit 20
```

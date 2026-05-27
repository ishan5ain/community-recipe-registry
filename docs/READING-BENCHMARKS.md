# How to Read Sparkrun Benchmarks

These benchmarks use [llama-benchy](https://github.com/pythagora-io/gptme/tree/main/src/llama_benchy) via sparkrun. Each result file comes in three formats (`.yaml`, `.json`, `.csv`) — all contain the same data.

## Two Phases of Inference

Every LLM request has two distinct phases:

```
Request arrives ──[PREFILL]──> first token out ──[DECODE]──> full response done
             ^ timing here         ^ TTFR          ^ tokens per second
```

- **Prefill**: The model reads your entire prompt in parallel. Heavy matrix multiply work — processes all input tokens at once.
- **Decode**: The model generates the response one token at a time, autoregressively. Each new token depends on all previous ones.

## Key Metrics

### Prefill Throughput (`pp t/s`) — "How fast does it read?"

Higher is better. This is raw GPU compute speed on the prompt. Longer prompts often get *higher* throughput because the GPU has more data to parallelize and saturates faster.

```
c=1 pp=512:  607 tokens/sec   (reads 512-token prompt in ~842ms)
c=1 pp=2048: 918 tokens/sec   (reads 2048-token prompt in ~2.2s)
```

### Decode Throughput (`tg t/s`) — "How fast does it write?"

This matters most for interactive use. Two numbers to watch:

- **Aggregate**: Total tokens/sec across all concurrent requests (server utilization)
- **Per-request**: `aggregate / concurrency` — what each individual user experiences

```
c=1:  17.5 t/s total → 17.5 per request   ← ideal for interactive chat
c=8:  96.1 t/s total →  12.0 per request  ← 8 users sharing the GPU
```

Rule of thumb: 20+ t/s feels instant. Below ~10 t/s starts feeling sluggish.

### Time to First Token (`TTFR`) — "How long until I see anything?"

User impatience metric. If TTFR exceeds ~3 seconds, users assume something is broken. Includes prefill time + queue wait from other requests ahead in line.

```
c=1  pp=512:   849 ms   ← snappy
c=1  pp=2048:  2,237 ms ← brief pause
c=8  pp=2048: 16,149 ms ← queue is the bottleneck
```

### Per-Request vs Aggregate Throughput

| Metric | Tells you... | Care about it when... |
|---|---|---|
| `tg t/s` (aggregate) | Server utilization | Capacity planning, total throughput |
| `tg t/s / concurrency` | Individual user experience | Latency SLAs, responsiveness |

## Concurrency Sweeps: What to Look For

When comparing across concurrency levels, watch for two patterns:

**Good scaling**: Aggregate decode scales near-linearly with `c`, per-request drops <30% from c=1. The GPU is efficiently batching work.

```
c=1 → 17 t/s/req   ← baseline
c=8 → 12 t/s/req   ← ~30% drop, acceptable
```

**Bad scaling**: Per-request drops >50% at moderate concurrency. Memory bandwidth is saturated or scheduler contention is high.

**Queueing problem**: TTFR grows faster than aggregate throughput improves. The server can't keep up with arriving requests.

## Practical Decision Guide

| If you care about... | Look at... | Target |
|---|---|---|
| Chat latency | c=1 decode + TTFR | >20 t/s, <2s TTFR |
| Batch processing | aggregate decode at max concurrency | maximize total t/s |
| Long prompt handling | prefill throughput | minimize prefill time |
| Capacity planning | per-request decode at target concurrency | find acceptable floor |

## File Format Details

Each benchmark run produces:

- `concurrency.json` — full breakdown per (depth, concurrency, prompt_size) combination
- `sweep.json` — same structure, different prompt length matrix
- `*.csv` — compact summary rows for spreadsheet use
- `*.yaml` — metadata + results in sparkrun format (includes recipe info, cluster config)

### JSON Structure

```json
{
  "model": "rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm",
  "benchmarks": [
    {
      "concurrency": 1,
      "prompt_size": 2048,
      "response_size": 32,
      "pp_throughput": { "mean": 918, "std": 21 },   // prefill t/s
      "tg_throughput": { "mean": 17.5, "std": 1.0 },  // decode t/s (aggregate)
      "ttfr": { "mean": 2237, "std": 51 }             // ms to first token
    }
  ]
}
```

The `values` array under each metric shows individual run measurements (usually 3–6 samples). The `mean` and `std` summarize them.

## Running Benchmarks Yourself

```bash
# Against a running server:
sparkrun benchmark run <recipe> \
  --skip-run \
  --no-stop \
  -H 127.0.0.1 \
  --solo \
  --port 8000 \
  -b served_model_name=<actual-model-id-on-server> \
  -b 'pp=512,2048' \
  -b 'concurrency=1,2,4,8'

# Flags:
#   --skip-run          Use already-running server instead of launching one
#   --no-stop           Don't stop the server after benchmarking
#   -b served_model_name Override model name (server may differ from recipe)
#   -b pp               Prompt lengths to test (comma-separated)
#   -b concurrency      Concurrency levels to test
```

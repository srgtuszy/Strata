# Community benchmark: Radeon AI PRO R9700, Ryzen 7 7700

Measured on 2026-10-02 by [srgtuszy](https://github.com/srgtuszy), on a Linux
machine called gigachad. This tests Strata 0.1.32 (HIP backend, source build)
with the original Flash-Next IQ3_S, one GPU, and a 32,768-token context limit.

Median decode throughput on the full (unpruned) IQ3_S model was **64.1 tok/s at
~3.2K prompt tokens and 69.4 tok/s at ~12.6K**, with prefill at **1,073** and
**1,327 tok/s** respectively. These are greedy summarization requests with a
256-token output cap; they do not establish answer quality. They were measured
without `STRATA_HIP_WMMA` (see "WMMA on this card" below).

## Hardware and software

- AMD Radeon AI PRO R9700 (Navi 48, gfx1201); 31.9 GiB (32,624 MiB) reported
  VRAM. PCIe Gen 5, x16 (32.0 GT/s, current and max). GPU clocks were not fixed.
- AMD Ryzen 7 7700, 8 cores / 16 threads, AVX-512. The engine selected
  **7 expert-pool workers + its host thread** (one per physical core, auto).
- 64 GiB installed DDR5, running at **EXPO 6000 MT/s** (DDR5-6000 CL30); Linux
  reported 61 GiB usable. Root LVM on NVMe.
- Ubuntu 24.04.4 LTS, kernel 7.0.0-34-generic.
- Source commit `c499bd102e7a4135c0de389dcfe38c399759ccc8`; engine **0.1.32**,
  locally compiled for `gfx1201` ("local-hip"). See [BUILD.json](BUILD.json).
- **System ROCm 7.2.4** in `/opt/rocm` (hipBLASLt 1.2.70204). No hipBLASLt
  tuning table matched (only a `1.5.0` table is shipped for gfx1201), so the
  prompt's dense matrix products ran on plain hipBLAS.
- The engine was compiled from source. **A local build patch was required on this
  distro:** ROCm 7.2.4's clang (22) did not find the system GCC C++ headers, so
  `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13` had to be appended to
  `CMAKE_HIP_FLAGS`. Without it `enable_language(HIP)` fails with
  `'cmath'/'cstdlib' file not found`. This is the same quirk we hit building
  upstream llama.cpp on this box; a `STRATA_HIP_GCC_DIR`-style hook in
  `setup.py` would make it self-service. (This is a report of what we ran; the
  patch is local and nothing here claims it belongs upstream.)

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`:

- `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` — 54,817,524,224 bytes
- `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` — 28,800,138,432 bytes

Locally computed SHA-256 (not cross-checked against the Hub's published LFS
hashes at measurement time): see [model-provenance.json](model-provenance.json).
MTP was fetched by the installer from `Qwen/Qwen3.8-Flash-Next`; its revision was
not recorded. The unmodified installer prepared the native IQ3_S pack and MTP
experts. The bundled default expert profile was used without calibration.

- Context 32,768; **INT8 KV**.
- Expert cache `auto`: **13,302 slots, 25.26 GiB VRAM**, profile-prefilled from
  the shipped 24,576-pair profile with **no eviction**. 592 MiB of VRAM free
  with everything loaded.
- Prefill `auto` selected 8,192-token chunks, borrowing 2,328 cache slots.
- MTP `--spec 4 --spec-min-p 0.5`; built-in suffix drafting enabled.
- Reasoning **on** (the engine's default); temperature 0; 256-token cap.
- Experimental speed projection off; no custom control vectors. Vision off (HIP).
- The complete measured configuration is [strata-iq3_s.json](strata-iq3_s.json);
  the startup + per-request log is [engine.log](engine.log).

## Method and reproduction

Prompts were **distinct** windows of `wikitext-2-raw-v1` (`wiki.test.raw`), so no
conversation prefix was reused; every run's engine log line shows `0 reused`
tokens. Each prompt appended `"\n\nSummarize the passage above in one sentence."`
and asked for up to 256 tokens, greedy (`temperature 0`). Two depths were tested:
~3,200 and ~12,600–13,500 prompt tokens (the exact counts differ per window).
Three runs per depth; both were measured with the same prompts.

Throughput is read from the engine's reported `timings` (and matches the
`strata serve: prompt N tokens = 0 reused + N read ...` log lines): prefill is
`prompt_per_second`, decode is `predicted_per_second`. No prefix reuse, no warm
conversation. The expert cache hit rate during decode was 97–98% throughout.

## Results — WMMA off (the configuration we run)

| depth | prompt tokens | prefill tok/s (3 runs) | median | decode tok/s (3 runs) | median |
|---|---|---|---|---|---|
| ~3.2K | 3,158 / 3,229 / 3,258 | 835.1 / 1073.4 / 1152.0 | **1073** | 63.6 / 71.0 / 64.1 | **64.1** |
| ~12.6K | 12,767 / 12,636 / 13,478 | 1312.6 / 1327.1 / 1388.7 | **1327** | 75.1 / 69.4 / 66.2 | **69.4** |

## Results — `STRATA_HIP_WMMA=1`

| depth | prompt tokens | prefill tok/s (3 runs) | median | decode tok/s (3 runs) | median |
|---|---|---|---|---|---|
| ~3.2K | 3,158 / 3,229 / 3,258 | 840.1 / 832.7 / 872.2 | **840** | 74.1 / 72.8 / 65.5 | **72.8** |
| ~12.6K | 12,767 / 12,636 / 13,478 | 1247.9 / 1246.9 / 1326.4 | **1247** | 81.7 / 75.7 / 63.9 | **75.7** |

## WMMA on this card

On this R9700 with the **full, unpruned IQ3_S**, `STRATA_HIP_WMMA=1` did **not**
improve prefill (medians 840 vs 1073 tok/s at 4K, 1247 vs 1327 at 16K) — it was
slightly slower, opposite to the Coder IQ1_M result in `docs/AMD_HIP.md`. Two
plausible reasons: the prompt path here is dominated by the expert gather, not
the QSA attention, so the matrix-core attention saves little; and WMMA changes
the prompt path's expert ring. `STRATA_PREFILL_RING=96` (recommended for the
16 GB 9070 XT) **hurt** 4K prefill on the R9700 further (948 tok/s, one run).
We run with WMMA **off**, which is also the deterministic path. Decode should be
unaffected by WMMA; the decode medians above differ mainly through MTP draft
acceptance (61–81% across runs, which dominates decode variance).

## Limits of this report

- Three runs per configuration; ranges overlap on decode (MTP acceptance varies
  run to run), so decode medians are indicative, not tight.
- 32K context only; contexts beyond 16K prompt tokens were not tested.
- No answer-quality or recall checks; this is a speed report only.
- Expert profile not calibrated; hipBLASLt not tuned; not an isolated OS.

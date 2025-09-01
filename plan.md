# Chatterbox – Inference & Performance Improvement Plan

This document outlines a prioritized plan to make TTS and VC inference faster, more memory‑efficient, and more robust on Windows, CPU, and GPU.

## Quick summary
- Disable unnecessary outputs in generation (attentions/hidden states) and avoid CFG by default → instant speed/memory win.
- Enable SDPA/Flash attention paths and AMP where safe (GPU) → big speedups for T3.
- Optimize S3Gen/HiFiGAN (channels_last, remove weight norm, AMP, compile) → faster vocoding.
- Cache conditionals and voice refs; make watermark optional → less repeated work.
- Add a tiny benchmark harness and knobs to compare toggles (RTF, tokens/s, mem).

---

## 1) Current pipeline and hotspots
- TTS pipeline (`ChatterboxTTS.generate`):
  1) Prepare conditionals (VoiceEncoder LSTM + S3Tokenizer + mel refs)
  2) T3 (LlamaModel-based) generates speech tokens with kv-cache, optional CFG (doubles batch)
  3) S3Gen converts tokens → mel (CFM) → waveform (HiFiGAN)
  4) Perth watermark applied on CPU array
- VC pipeline (`ChatterboxVC.generate`): tokenize source audio with S3Tokenizer, synth with S3Gen using reference dict

Likely bottlenecks:
- T3 generation loop sets `output_attentions=True` and `output_hidden_states=True` (extra compute/memory every step)
- CFG doubles compute when enabled
- No AMP on GPU for T3 or S3Gen; no SDPA toggles
- HiFiGAN keeps weight norm at inference; not channels_last
- Repeated conditional prep for the same reference file; watermark always applied
- Librosa I/O on CPU for resampling; mixed backends

---

## 2) Immediate low‑risk wins (P0)
1) Turn off extra outputs in the generation loop
   - In `T3.inference`, call `self.patched_model(..., output_attentions=False, output_hidden_states=False)` for both the first and step calls.
2) Default to CFG off (or lower)
   - Keep `cfg_weight` default at 0.0 in examples/UI for speed. Users can opt in.
3) Make watermark optional
   - Add a `watermark=True` flag at class or `generate()` level to skip the `perth` call for speed comparisons.
4) Prefer torchaudio I/O
   - Replace `librosa.load` with `torchaudio.load` + `Resample` where possible for consistency/perf on Windows.

Expected impact: noticeable speed/memory reduction with minimal risk.

---

## 3) Transformer (T3) speedups (P0–P1)
- Enable fused attention paths
  - Set at init: `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.set_float32_matmul_precision("high")`.
  - Prefer PyTorch SDPA (default in 2.6); ensure `torch.backends.cuda.enable_flash_sdp(True)` when supported.
- Mixed precision on GPU
  - Wrap T3 forward/generation in `torch.autocast(device_type="cuda", dtype=torch.bfloat16 or torch.float16)` where numerically safe.
- KV cache and model flags
  - Already using kv-cache; removing extra outputs reduces its overhead.
- Optional compile
  - Try `torch.compile(self.patched_model.model, mode="max-autotune")` with a short warm up; fall back gracefully if dynamic shapes block speedups. Keep disabled by default on Windows until validated.
- Quantization tracks (optional/experimental)
  - Bitsandbytes 8‑bit/4‑bit loading path (requires refactor to use HF `LlamaForCausalLM` or custom modules). Gate behind a flag.

Expected impact: 1.3–2.0× on modern NVIDIA GPUs; lower on CPUs/Windows without flash.

---

## 4) S3Gen + HiFiGAN (decoder) speedups (P0–P1)
- Memory format
  - On CUDA set all conv-heavy modules to channels_last: `module = module.to(memory_format=torch.channels_last)` and input tensors likewise.
- Remove weight norm at inference
  - Call `HiFTGenerator.remove_weight_norm()` once after load.
- AMP on GPU
  - Wrap S3Gen forward in autocast; verify output quality is unchanged.
- Optional compile
  - `torch.compile` the S3Gen/HiFiGAN forward with `mode="max-autotune"` after a fixed‑shape warm‑up; cache compiled modules.
- Streaming/low‑latency path (P1)
  - Use `S3Token2Wav.flow_inference(..., finalize=False)` in a loop, pass `cache_source` to `hift_inference` to reduce latency and reuse source cache. Expose a streaming API that yields audio chunks.

Expected impact: 1.2–1.7× speedup; much lower initial latency with streaming.

---

## 5) VoiceEncoder & conditionals (P0)
- Cache conditionals
  - Memoize `prepare_conditionals(audio_prompt_path)` by file path + mtime/hash; reuse across calls.
- VoiceEncoder optimizations
  - On CPU: dynamic quantization for LSTM + `torch.compile` (already attempted) but scope it only to VE if whole‑model fails.
  - Ensure `flatten_parameters()` is called once (config option exists) and that batch inference uses reasonable chunking.

---

## 6) I/O, resampling, and preprocessing (P0)
- Use `torchaudio.load` and `torchaudio.functional.resample` or `ta.transforms.Resample` (already partially cached via `lru_cache`).
- Keep tensors on device to avoid host<->device syncs; use pinned memory for host‑side buffers.
- Remove redundant conversions (`numpy`↔`torch`) in hot paths where possible.

---

## 7) Controls, configs, and UX (P0)
- Add a runtime config object (env vars or flags) for:
  - enable_amp (gpu), enable_compile, enable_channels_last, enable_remove_wn, enable_watermark, enable_sdpa_flags
- Expose in Gradio: Performance preset (Fast/Quality/Experimental)
  - Fast: no CFG, AMP on, SDPA/TF32 on, HiFiGAN WN removed
  - Quality: CFG on (optional), AMP off
  - Experimental: compile on

---

## 8) Benchmarks and quality gates (P0)
Add a tiny benchmark script (not yet implemented) to report:
- T3 tokens/s (with/without CFG; with/without AMP)
- Decoder RTF (seconds of audio per second compute)
- End‑to‑end latency for a fixed prompt
- GPU mem peak and CPU mem usage (if available)

Acceptance criteria:
- P0 (no quality loss):
  - ≥25% faster T3 tokens/s (attn/hidden states off, AMP+SDPA on)
  - ≥15% faster decoder RTF (channels_last + remove WN + AMP)
- P1 (optional):
  - Further ≥10–20% via compile on supported setups

---

## 9) Windows specifics and caveats
- Prefer PyTorch SDPA over third‑party FlashAttention on Windows.
- `torch.compile` may provide smaller gains; keep optional and well‑guarded.
- Use `torchaudio` for I/O; avoid `librosa` resampling in hot paths.
- Ensure Visual C++ Redistributable and latest GPU driver; CUDA 12.x recommended.

---

## 10) Implementation roadmap
- P0 (day 1–2)
  - T3: disable extra outputs; set SDPA/TF32 flags; wire AMP path; default CFG off
  - S3Gen/HiFiGAN: channels_last, remove WN, AMP path
  - Add watermark flag; torchaudio I/O; conditional caching
  - Basic benchmark script and readme notes
- P1 (day 3–5)
  - Optional `torch.compile` for S3Gen/HiFiGAN; warm‑ups and fallbacks
  - Streaming decoder API using `cache_source`
- P2 (week 2+)
  - Quantized T3 path (8‑bit/4‑bit) or torchao float‑8 (hardware‑dependent)
  - Deeper batching/micro‑batching support for multi‑utterance queues

---

## 11) Concrete code changes (overview)
- `src/chatterbox/models/t3/t3.py`
  - In `inference()`: set `output_attentions=False`, `output_hidden_states=False`
  - Add optional AMP context (GPU only)
  - Add SDPA/TF32 enable helper at init
- `src/chatterbox/models/s3gen/s3gen.py` and HiFiGAN
  - Expose `remove_weight_norm()` call after load
  - Optionally set modules to channels_last; add AMP context
- `src/chatterbox/tts.py` and `src/chatterbox/vc.py`
  - Add `watermark` flag; cache conditionals by file path
  - Replace `librosa` I/O with `torchaudio` where feasible
- Gradio apps
  - Add Performance preset toggle and show measured timings
- New: `scripts/bench.py` to measure RTF/tokens‑per‑sec and log environment

---

## 12) Risks
- AMP may alter numerical behavior slightly; gate behind flag and test quality
- `torch.compile` can regress or fail with dynamic shapes; keep opt‑in with fallbacks
- 4‑bit/8‑bit quantization needs extra dependencies and careful integration

---

## 13) Definition of done
- P0 toggles landed, unit smoke tests pass, benchmark shows target improvements on at least one NVIDIA GPU + Windows; no regression in audio quality for default settings.

Chatterbox — AGENTS Guide

Purpose

- Give AI agents and contributors a fast, reliable way to work within this repo.
- Explain the architecture, key entry points, how to run things, and common pitfalls.
- Define expectations for changes, code style, and dependency constraints.

Repo Overview

- Type: Python package (`chatterbox-tts`) with examples and Gradio demos
- Capabilities:
  - English TTS: `ChatterboxTTS`
  - Multilingual TTS (23 languages): `ChatterboxMultilingualTTS`
  - Voice Conversion (VC): `ChatterboxVC`
- Model weights: Fetched at runtime from Hugging Face Hub; nothing large checked into repo

Key Entry Points

- English TTS: `src/chatterbox/tts.py`
- Multilingual TTS: `src/chatterbox/mtl_tts.py`
- Voice Conversion: `src/chatterbox/vc.py`
- Core models:
  - Token-to-token model (text→speech tokens): `src/chatterbox/models/t3/t3.py`
  - Speech generator (tokens→waveform): `src/chatterbox/models/s3gen/s3gen.py`
  - Speech tokenizer (16 kHz): `src/chatterbox/models/s3tokenizer/s3tokenizer.py`
  - Voice encoder: `src/chatterbox/models/voice_encoder/voice_encoder.py`
- Examples & demos:
  - CLI examples: `example_tts.py`, `example_vc.py`, `example_for_mac.py`
  - Gradio English TTS: `gradio_tts_app.py`
  - Gradio Multilingual TTS: `multilingual_app.py`
- Packaging/config: `pyproject.toml`

How Things Work (High Level)

- Pipeline (TTS):
  1) Normalize input text and tokenize with language-appropriate tokenizer
  2) T3 model predicts speech token sequence conditioned on reference voice/style
  3) S3Gen turns speech tokens into waveform at 24 kHz, applies watermark
  4) Output saved or returned as tensor
- Conditioning:
  - Reference audio is embedded into conditionals (speaker embedding, prompt tokens, mel features)
  - Exaggeration controls expressiveness; CFG weight affects guidance/pace
- Watermarking:
  - Every output is watermarked via `resemble-perth` and should remain enabled

Runtime Requirements

- Python: >= 3.10 (developed on 3.11)
- PyTorch: pinned in `pyproject.toml` (torch==2.6.0, torchaudio==2.6.0)
- GPU strongly recommended; CPU works but is slow
- macOS (Apple Silicon): use MPS when available; see `example_for_mac.py`
- Hugging Face Hub network access is required to fetch weights on first run
- Multilingual model uses `snapshot_download` and can accept `HF_TOKEN` for auth/rate limits

Install & Quick Start

- Editable install:
  - `pip install -e .`
- English TTS example:
  - `python example_tts.py` (generates `test-1.wav`)
- Voice Conversion example:
  - `python example_vc.py` (set `AUDIO_PATH` and `TARGET_VOICE_PATH` first)
- Gradio demos:
  - English: `python gradio_tts_app.py`
  - Multilingual: `python multilingual_app.py`

Public API Surfaces

- English TTS (`src/chatterbox/tts.py`):
  - `ChatterboxTTS.from_pretrained(device)` → downloads weights; returns model
  - `model.generate(text, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, min_p=0.05, top_p=1.0, repetition_penalty=1.2)` → `torch.Tensor [1, L]`
- Multilingual TTS (`src/chatterbox/mtl_tts.py`):
  - `ChatterboxMultilingualTTS.from_pretrained(device)`
  - `model.generate(text, language_id, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, min_p=0.05, top_p=1.0, repetition_penalty=2.0)`
  - `ChatterboxMultilingualTTS.get_supported_languages()` → dict of codes→names
- Voice Conversion (`src/chatterbox/vc.py`):
  - `ChatterboxVC.from_pretrained(device)`
  - `model.generate(audio, target_voice_path=None)`

Important Constants and Sample Rates

- S3GEN output sample rate: `src/chatterbox/models/s3gen/const.py`
- S3Tokenizer input sample rate: `src/chatterbox/models/s3tokenizer/s3tokenizer.py`
- TTS classes export `model.sr` (use this when saving with torchaudio)

Gradio Demos

- English app: loads `ChatterboxTTS` and exposes sliders for exaggeration/temperature/CFG/min_p/top_p/repetition_penalty
- Multilingual app: wraps `ChatterboxMultilingualTTS`, provides language selection, optional default per-language audio prompts, and the same core controls

Models and Weights

- English TTS/VC use `hf_hub_download` for files like `ve.safetensors`, `t3_cfg.safetensors`, `s3gen.safetensors`, `tokenizer.json`, `conds.pt`
- Multilingual TTS uses `snapshot_download` with allow_patterns for multilingual assets; supports `HF_TOKEN` env var
- Do not add large model files to the repo; always fetch from HF Hub

Gotchas and Best Practices

- Device selection:
  - Prefer CUDA if available; otherwise use MPS on macOS; else CPU
  - `from_pretrained` in English classes auto-detects MPS availability; falls back to CPU
- Punctuation normalization:
  - Both TTS classes normalize punctuation before tokenization; ensure text is not empty
- Exaggeration and CFG:
  - High `exaggeration` increases expressiveness but may speed speech; reduce `cfg_weight` for pacing
  - For language transfer with multilingual, set `cfg_weight=0` to avoid accent bleed from prompt
- Lengths and truncation:
  - Reference audio for conditioning is truncated internally (~10 seconds) to control compute
- Watermarking:
  - Watermark is always applied before returning audio; do not remove without explicit request
- Multilingual specifics:
  - `language_id` must be in supported set; see `ChatterboxMultilingualTTS.get_supported_languages()`
- Environment stability:
  - Dependencies are pinned; avoid changing versions unless you know the downstream effect

Code Style and Change Guidelines

- Keep changes minimal and scoped; prefer surgical edits
- Python style: standard PEP 8; follow existing naming patterns
- Don’t change public method signatures in `ChatterboxTTS`, `ChatterboxMultilingualTTS`, or `ChatterboxVC` unless necessary
- If adding parameters, provide sensible defaults to preserve backward compatibility
- Maintain device handling; any new tensors should respect the model’s device
- Keep sample rate handling consistent; save audio using `model.sr`
- Leave watermarking on by default for any new generation paths

Common Tasks (Playbooks)

- Add a new control knob to English Gradio demo:
  - Update `gradio_tts_app.py` inputs/slider
  - Pass the new value through to `model.generate(...)`
- Add/adjust default prompts in multilingual demo:
  - Edit `LANGUAGE_CONFIG` in `multilingual_app.py` to tweak default audio/text per language
- Add a smoke test script:
  - Create a small script that loads a tiny text and runs `from_pretrained(...).generate(...)` on CPU to confirm no import/runtime errors
- Update dependency pinning:
  - Change `pyproject.toml`, then verify English TTS and VC examples run locally

Validation Checklist (before PR or handoff)

- English TTS: run `example_tts.py` on your device; confirm `test-1.wav` plays and is watermarked
- Multilingual TTS: run `multilingual_app.py`; synthesize at least one non-English language
- VC: run `example_vc.py` with small inputs; confirm output generated
- Gradio demos: confirm UI loads and inference works once per change touching demos

Security/Compliance Notes

- Watermarking must remain enabled by default in all generation paths
- Do not embed secrets; rely on env vars (e.g., `HF_TOKEN`) where needed

Folder Structure (selected)

- `src/chatterbox/...` — library code and models
- `example_*.py` — quick runnable examples
- `gradio_*.py`, `multilingual_app.py` — Gradio demos
- `pyproject.toml` — package metadata and pinned deps

Scope and Precedence

- This AGENTS.md applies to the entire repository
- If additional AGENTS.md files are added deeper in subfolders, those take precedence for their subtree
- Direct user instructions override this document

Contact & Attribution

- Original authors: Resemble AI
- License: MIT (`LICENSE`)

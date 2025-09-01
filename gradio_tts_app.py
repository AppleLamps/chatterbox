import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
import functools
import threading
import os
import warnings

# Reduce warning noise from transformers and other libraries
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gradio.processing_utils")
warnings.filterwarnings("ignore", category=UserWarning, module="perth.perth_net")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers.models.lora")
warnings.filterwarnings("ignore", category=UserWarning, message=".*flash attention.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*past_key_values.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*LlamaModel.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.backends.cuda.sdp_kernel.*")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model cache - lazy loaded and shared across all sessions
_model_cache = None
_model_lock = threading.Lock()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_cached_model():
    """Global lazy-loaded model that persists across all sessions"""
    global _model_cache
    if _model_cache is None:
        with _model_lock:
            if _model_cache is None:  # Double-check locking pattern
                print(f"Loading ChatterboxTTS model on {DEVICE}...")
                _model_cache = ChatterboxTTS.from_pretrained(DEVICE)
                print("Model loaded successfully!")
    return _model_cache


def load_model():
    """Session loader - returns the globally cached model"""
    return get_cached_model()


def generate_with_status(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    """Generate audio with status updates"""
    # Use globally cached model - no need to reload per session
    if model is None:
        model = get_cached_model()

    if seed_num != 0:
        set_seed(int(seed_num))

    # Generate status message
    status_msg = f"Generating... (Device: {DEVICE}"
    if audio_prompt_path:
        status_msg += ", Using voice reference"
    status_msg += ")"
    
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    
    # Convert to proper format to avoid Gradio warnings
    wav_numpy = wav.squeeze(0).numpy()
    # Ensure proper scaling and clipping for audio output
    wav_numpy = np.clip(wav_numpy, -1.0, 1.0)
    
    final_status = f"Generated successfully! (Model cached on {DEVICE})"
    return (model.sr, wav_numpy), final_status


def apply_preset(preset_name, current_exag, current_cfg):
    """Apply parameter presets based on README recommendations"""
    if preset_name == "General Use (Default)":
        return 0.5, 0.5
    elif preset_name == "Expressive/Dramatic":
        return 0.7, 0.3
    elif preset_name == "Fast Speaker":
        return current_exag, 0.3
    else:  # Custom
        return current_exag, current_cfg


with gr.Blocks(title="Chatterbox TTS - Optimized") as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize",
                max_lines=5
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            
            # Preset selector based on README tips
            preset_dropdown = gr.Dropdown(
                choices=["General Use (Default)", "Expressive/Dramatic", "Fast Speaker", "Custom"],
                value="General Use (Default)",
                label="Parameter Presets (Based on README recommendations)"
            )
            
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=0.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace (Lower = faster speech, Higher = slower/more deliberate)", value=0.5)

            with gr.Accordion("Advanced Parameters", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature (Controls randomness)", value=0.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p (Recommended: 0.02-0.1, handles higher temperatures better)", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p (Original sampler, 1.0 recommended to disable)", value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.01, label="Repetition penalty", value=1.2)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")
            
            # Status indicator
            status_text = gr.Textbox(
                value="Ready to generate (Model will load on first use)",
                label="Status",
                interactive=False
            )

    # Initialize with cached model on app start (lazy loading)
    demo.load(fn=load_model, inputs=[], outputs=model_state)
    
    # Preset change handler
    preset_dropdown.change(
        fn=apply_preset,
        inputs=[preset_dropdown, exaggeration, cfg_weight],
        outputs=[exaggeration, cfg_weight]
    )

    run_btn.click(
        fn=generate_with_status,
        inputs=[
            model_state,
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=[audio_output, status_text],
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)

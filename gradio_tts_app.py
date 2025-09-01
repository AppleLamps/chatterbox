import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
import functools
import threading
import os
import warnings
import time

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


def get_cached_model(progress=None):
    """Global lazy-loaded model that persists across all sessions"""
    global _model_cache
    if _model_cache is None:
        with _model_lock:
            if _model_cache is None:  # Double-check locking pattern
                if progress:
                    progress(0.1, desc="Initializing Chatterbox model...")
                print(f"Loading ChatterboxTTS model on {DEVICE}...")
                
                if progress:
                    progress(0.3, desc="Downloading model files...")
                # The actual model loading happens here
                _model_cache = ChatterboxTTS.from_pretrained(DEVICE)
                
                if progress:
                    progress(0.8, desc="Finalizing model setup...")
                    time.sleep(0.5)  # Small delay for visual feedback
                    progress(1.0, desc="Model loaded successfully!")
                    
                print("Model loaded successfully!")
    return _model_cache


def load_model_with_progress(progress=gr.Progress()):
    """Session loader with progress tracking - returns the globally cached model"""
    progress(0.0, desc="Checking model cache...")
    model = get_cached_model(progress)
    return model, "Model loaded successfully! Ready to generate."


def generate_with_status(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty, progress=gr.Progress()):
    """Generate audio with detailed status updates and progress tracking"""
    progress(0.0, desc="🚀 Initializing generation...")
    
    # Use globally cached model - no need to reload per session
    if model is None:
        progress(0.05, desc="⚠️ Loading model...")
        model = get_cached_model()

    progress(0.1, desc="🔧 Preparing generation parameters...")
    if seed_num != 0:
        set_seed(int(seed_num))
        progress(0.15, desc=f"🎲 Set random seed to {int(seed_num)}")

    # Generate initial status message
    status_msg = f"Preparing generation (Device: {DEVICE}"
    if audio_prompt_path:
        status_msg += ", Using voice reference"
    status_msg += ")"
    
    progress(0.2, desc="📝 Processing text input...")
    text_length = len(text.split())
    
    # If using audio prompt, prepare conditionals
    if audio_prompt_path:
        progress(0.3, desc="🎤 Analyzing voice reference...")
        time.sleep(0.3)  # Small delay for visual feedback
    else:
        progress(0.3, desc="🔊 Using default voice model...")
    
    progress(0.4, desc="🎭 Configuring voice parameters...")
    progress(0.45, desc=f"📊 Text length: {text_length} words")
    
    # Update status for generation phase
    generation_status = f"🎵 Synthesizing audio... (Device: {DEVICE}"
    if audio_prompt_path:
        generation_status += ", Using voice reference"
    generation_status += ")"
    
    progress(0.5, desc="🧠 Running T3 text-to-speech model...")
    generation_start_time = time.time()
    
    # Generate the audio
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
    
    generation_time = time.time() - generation_start_time
    
    progress(0.8, desc="🔊 Processing generated audio...")
    
    # Convert to proper format to avoid Gradio warnings
    wav_numpy = wav.squeeze(0).numpy()
    # Ensure proper scaling and clipping for audio output
    wav_numpy = np.clip(wav_numpy, -1.0, 1.0)
    
    audio_duration = len(wav_numpy) / model.sr
    
    progress(0.9, desc="✨ Finalizing output...")
    time.sleep(0.1)  # Small delay for visual feedback
    
    progress(1.0, desc="✅ Generation complete!")
    
    final_status = f"""✅ **Generation Successful!**
📊 Audio Duration: {audio_duration:.2f}s
⏱️ Generation Time: {generation_time:.2f}s
📝 Words Processed: {text_length}
🖥️ Device: {DEVICE}
{'🎤 Voice Reference: Used' if audio_prompt_path else '🔊 Voice Reference: Default'}"""
    
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


def check_system_status():
    """Return system information for debugging"""
    return f"""
🖥️ **System Information:**
- Device: {DEVICE}
- PyTorch version: {torch.__version__}
- CUDA available: {torch.cuda.is_available()}
- Model cached: {"✅ Yes" if _model_cache is not None else "❌ No"}
"""


def validate_model_loaded(model):
    """Check if model is loaded before generation"""
    if model is None:
        return False, "⚠️ Please load the model first by clicking '🚀 Load Model' button"
    return True, "✅ Model ready for generation"


def generate_with_validation(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty, progress=gr.Progress()):
    """Validate model is loaded before generation"""
    is_loaded, message = validate_model_loaded(model)
    if not is_loaded:
        return None, message
    
    return generate_with_status(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty, progress)


with gr.Blocks(title="Chatterbox TTS - Enhanced with Progress Tracking") as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🎤 Chatterbox Text-to-Speech")
            gr.Markdown("Advanced neural TTS with real-time progress tracking")
            
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize",
                max_lines=5,
                placeholder="Enter the text you want to convert to speech..."
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File (Optional)", value=None)
            
            # Preset selector based on README tips
            preset_dropdown = gr.Dropdown(
                choices=["General Use (Default)", "Expressive/Dramatic", "Fast Speaker", "Custom"],
                value="General Use (Default)",
                label="🎯 Parameter Presets (Based on README recommendations)"
            )
            
            exaggeration = gr.Slider(0.25, 2, step=.05, label="🎭 Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=0.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="⚡ CFG/Pace (Lower = faster speech, Higher = slower/more deliberate)", value=0.5)

            with gr.Accordion("🔧 Advanced Parameters", open=False):
                seed_num = gr.Number(value=0, label="🎲 Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="🌡️ Temperature (Controls randomness)", value=0.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="📊 min_p (Recommended: 0.02-0.1, handles higher temperatures better)", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="🔝 top_p (Original sampler, 1.0 recommended to disable)", value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.01, label="🔄 Repetition penalty", value=1.2)

            with gr.Row():
                load_btn = gr.Button("🚀 Load Model", variant="secondary", scale=1)
                run_btn = gr.Button("🎵 Generate Audio", variant="primary", scale=2)

        with gr.Column():
            audio_output = gr.Audio(label="🔊 Generated Audio")
            
            # Status indicator with better formatting
            status_text = gr.Textbox(
                value="👋 Welcome! Click '🚀 Load Model' to initialize Chatterbox TTS",
                label="📊 Status & Progress",
                interactive=False,
                lines=2
            )
            
            # Additional info panel
            with gr.Accordion("ℹ️ Generation Info", open=False):
                gr.Markdown("""
                **Tips for better results:**
                - Use clear, well-punctuated text
                - Reference audio should be 3-10 seconds long
                - Experiment with exaggeration values between 0.3-0.8
                - Lower CFG values create faster speech
                """)
                
            with gr.Accordion("🔧 System Status", open=False):
                system_status = gr.Markdown(check_system_status())
                refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")

    # Load model button click handler
    load_btn.click(
        fn=load_model_with_progress,
        inputs=[],
        outputs=[model_state, status_text]
    )
    
    # Refresh system status button
    refresh_status_btn.click(
        fn=check_system_status,
        inputs=[],
        outputs=[system_status]
    )
    
    # Initialize with cached model on app start (lazy loading) - removed automatic loading
    # Users now click "Load Model" button for explicit control
    
    # Preset change handler
    preset_dropdown.change(
        fn=apply_preset,
        inputs=[preset_dropdown, exaggeration, cfg_weight],
        outputs=[exaggeration, cfg_weight]
    )

    run_btn.click(
        fn=generate_with_validation,
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

import os
import sys
import gradio as gr
import torch
import numpy as np
from pathlib import Path
from pydub import AudioSegment  # ✅ added

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from inference import WhisperInference
from utils import format_time


MODEL_PATH = os.environ.get('MODEL_PATH', '../models/final_model')
BASE_MODEL = os.environ.get('BASE_MODEL', 'openai/whisper-small')


# ✅ Helper to convert MP3 → WAV safely
def ensure_wav(file_path):
    """Convert .mp3 to .wav if needed and return the .wav path"""
    if file_path and file_path.endswith(".mp3"):
        try:
            wav_path = file_path.replace(".mp3", ".wav")
            sound = AudioSegment.from_file(file_path)
            sound.export(wav_path, format="wav")
            print(f"[Audio Conversion] Converted MP3 → WAV: {wav_path}")
            return wav_path
        except Exception as e:
            print(f"[Audio Conversion Error] Failed to convert {file_path}: {e}")
            return file_path
    return file_path


class GradioASRDemo:
    def __init__(self, model_path: str, base_model: str):
        self.model_path = model_path
        self.base_model = base_model

        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist.")
            print(f"Using base model {base_model} instead.")
            self.model_path = base_model

        print(f"Loading model from: {self.model_path}")
        self.inferencer = WhisperInference(
            model_path=self.model_path,
            base_model=self.base_model,
            use_onnx=False
        )
        print("Model loaded successfully!")

    def transcribe_audio(self, audio_input, language=None):
        if audio_input is None:
            return "No audio provided.", "", ""

        try:
            if isinstance(audio_input, tuple):
                sample_rate, audio_data = audio_input

                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0

                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)

                if sample_rate != 16000:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

                audio_tensor = torch.from_numpy(audio_data)
            else:
                audio_tensor = audio_input

            result = self.inferencer.transcribe(
                audio_tensor,
                language=language if language and language != "auto" else None
            )

            transcription = result['transcription']
            processing_time = result['processing_time']
            audio_duration = result['audio_duration']
            rtf = result['rtf']

            stats = f"""
**Performance Metrics:**
- Audio Duration: {audio_duration:.2f}s
- Processing Time: {processing_time:.3f}s
- Real-Time Factor (RTF): {rtf:.3f}
- Status: {'✓ Real-time capable' if rtf <= 0.5 else '⚠ Not real-time'}
            """

            rtf_info = f"RTF: {rtf:.3f} (Target: ≤ 0.5)"

            return transcription, stats, rtf_info

        except Exception as e:
            return f"Error: {str(e)}", "", ""

    def transcribe_file(self, file_path, language=None):
        if file_path is None:
            return "No file provided.", "", ""

        # ✅ Convert MP3 → WAV before passing to Whisper
        file_path = ensure_wav(file_path)

        try:
            result = self.inferencer.transcribe(
                file_path,
                language=language if language and language != "auto" else None
            )

            transcription = result['transcription']
            processing_time = result['processing_time']
            audio_duration = result['audio_duration']
            rtf = result['rtf']

            stats = f"""
**Performance Metrics:**
- Audio Duration: {audio_duration:.2f}s
- Processing Time: {processing_time:.3f}s
- Real-Time Factor (RTF): {rtf:.3f}
- Status: {'✓ Real-time capable' if rtf <= 0.5 else '⚠ Not real-time'}
            """

            rtf_info = f"RTF: {rtf:.3f} (Target: ≤ 0.5)"

            return transcription, stats, rtf_info

        except Exception as e:
            return f"Error: {str(e)}", "", ""


def create_demo(model_path: str, base_model: str):
    demo_app = GradioASRDemo(model_path, base_model)

    with gr.Blocks(title="Accent-Adaptive ASR System", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎙️ Accent-Adaptive ASR System

            Advanced Automatic Speech Recognition system fine-tuned for underrepresented accents and dialects.

            **Features:**
            - Low-resource accent adaptation using LoRA fine-tuning
            - Real-time transcription (RTF ≤ 0.5)
            - Multi-language support
            - High accuracy with minimal training data
            """
        )

        with gr.Tabs():
            with gr.Tab("🎤 Microphone Recording"):
                with gr.Row():
                    with gr.Column():
                        mic_input = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="Record Audio"
                        )
                        mic_language = gr.Dropdown(
                            choices=["auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"],
                            value="auto",
                            label="Language (optional)"
                        )
                        mic_button = gr.Button("Transcribe", variant="primary")

                    with gr.Column():
                        mic_output = gr.Textbox(
                            label="Transcription",
                            placeholder="Your transcription will appear here...",
                            lines=5
                        )
                        mic_stats = gr.Markdown(label="Statistics")
                        mic_rtf = gr.Textbox(label="Real-Time Factor", interactive=False)

                mic_button.click(
                    fn=demo_app.transcribe_audio,
                    inputs=[mic_input, mic_language],
                    outputs=[mic_output, mic_stats, mic_rtf]
                )

            with gr.Tab("📁 File Upload"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.Audio(
                            sources=["upload"],
                            type="filepath",
                            label="Upload Audio File"
                        )
                        file_language = gr.Dropdown(
                            choices=["auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"],
                            value="auto",
                            label="Language (optional)"
                        )
                        file_button = gr.Button("Transcribe", variant="primary")

                    with gr.Column():
                        file_output = gr.Textbox(
                            label="Transcription",
                            placeholder="Your transcription will appear here...",
                            lines=5
                        )
                        file_stats = gr.Markdown(label="Statistics")
                        file_rtf = gr.Textbox(label="Real-Time Factor", interactive=False)

                file_button.click(
                    fn=demo_app.transcribe_file,
                    inputs=[file_input, file_language],
                    outputs=[file_output, file_stats, file_rtf]
                )

        gr.Markdown(
            """
            ---
            ### 📊 About This System

            This ASR system uses:
            - **Base Model**: Whisper (Small) - Pretrained multilingual speech recognition
            - **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient accent adaptation
            - **Augmentation**: Comprehensive pipeline including speed, pitch, noise, and reverb
            - **Target**: WER reduction ≥ 30% on low-resource accents

            **Real-Time Factor (RTF)**: Ratio of processing time to audio duration.
            RTF ≤ 0.5 indicates real-time capability (processes faster than playback speed).

            ### 🎯 Performance Goals
            - ✓ WER Reduction: ≥ 30% vs baseline
            - ✓ RTF: ≤ 0.5 (real-time capable)
            - ✓ Training: < 1 hour on small datasets
            """
        )

    return demo


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Launch Gradio ASR Demo')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--base_model', type=str, default=BASE_MODEL)
    parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    parser.add_argument('--server_name', type=str, default='0.0.0.0')
    parser.add_argument('--server_port', type=int, default=7860)

    args = parser.parse_args()

    demo = create_demo(args.model_path, args.base_model)

    print("\n" + "="*60)
    print("🎙️  Accent-Adaptive ASR Demo")
    print("="*60)
    print(f"Model Path: {args.model_path}")
    print(f"Base Model: {args.base_model}")
    print(f"Share: {args.share}")
    print("="*60 + "\n")

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port
    )


if __name__ == '__main__':
    main()

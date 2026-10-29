import os
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict
import time

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from utils import setup_logging, get_device, Timer, calculate_rtf


class WhisperInference:
    def __init__(
        self,
        model_path: str,
        base_model: str = "openai/whisper-small",
        use_onnx: bool = False,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.use_onnx = use_onnx

        if device:
            self.device = torch.device(device)
        else:
            self.device = get_device()

        self.logger = setup_logging()
        self.logger.info(f"Initializing inference on {self.device}")

        self._load_model()

    def _load_model(self):
        self.logger.info(f"Loading processor from {self.base_model}")
        self.processor = WhisperProcessor.from_pretrained(self.base_model)

        if os.path.exists(os.path.join(self.model_path, 'adapter_config.json')):
            self.logger.info(f"Loading LoRA model from {self.model_path}")

            base = WhisperForConditionalGeneration.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )

            self.model = PeftModel.from_pretrained(base, self.model_path)
            self.model = self.model.merge_and_unload()
        else:
            self.logger.info(f"Loading full model from {self.model_path}")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )

        self.model.to(self.device)
        self.model.eval()

        if self.use_onnx:
            try:
                self._export_to_onnx()
            except Exception as e:
                self.logger.warning(f"ONNX export failed: {e}. Using PyTorch inference.")
                self.use_onnx = False

    def _export_to_onnx(self):
        import onnxruntime as ort

        onnx_path = os.path.join(self.model_path, 'model.onnx')

        if not os.path.exists(onnx_path):
            self.logger.info("Exporting model to ONNX format...")

            dummy_input = torch.randn(1, 80, 3000).to(self.device)

            torch.onnx.export(
                self.model.model.encoder,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input_features'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'input_features': {0: 'batch_size', 2: 'sequence_length'},
                    'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
                }
            )

            self.logger.info(f"Model exported to {onnx_path}")

        self.logger.info("Loading ONNX runtime session")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)

    def load_audio(self, audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)

        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform.squeeze(0)

    def transcribe(
        self,
        audio_input: Union[str, torch.Tensor, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        if isinstance(audio_input, str):
            audio = self.load_audio(audio_input)
            audio_duration = audio.shape[0] / 16000
        elif isinstance(audio_input, np.ndarray):
            audio = torch.from_numpy(audio_input)
            audio_duration = len(audio_input) / 16000
        else:
            audio = audio_input
            audio_duration = audio.shape[0] / 16000

        with Timer() as timer:
            input_features = self.processor(
                audio.numpy() if isinstance(audio, torch.Tensor) else audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features

            input_features = input_features.to(self.device)

            with torch.no_grad():
                if self.use_onnx and hasattr(self, 'ort_session'):
                    encoder_outputs = self.ort_session.run(
                        None,
                        {'input_features': input_features.cpu().numpy()}
                    )

                    predicted_ids = self.model.generate(
                        encoder_outputs=torch.from_numpy(encoder_outputs[0]).to(self.device)
                    )
                else:
                    forced_decoder_ids = None
                    if language:
                        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                            language=language,
                            task=task
                        )

                    predicted_ids = self.model.generate(
                        input_features,
                        forced_decoder_ids=forced_decoder_ids
                    )

            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

        processing_time = timer.elapsed
        rtf = calculate_rtf(audio_duration, processing_time)

        return {
            'transcription': transcription,
            'processing_time': processing_time,
            'audio_duration': audio_duration,
            'rtf': rtf
        }

    def transcribe_batch(self, audio_paths: list, language: Optional[str] = None) -> list:
        results = []

        for audio_path in audio_paths:
            self.logger.info(f"Processing: {audio_path}")
            result = self.transcribe(audio_path, language=language)
            result['audio_path'] = audio_path
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(description='Whisper Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to fine-tuned model')
    parser.add_argument('--base_model', type=str, default='openai/whisper-small')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to audio file or directory')
    parser.add_argument('--output_file', type=str, default='transcriptions.txt')
    parser.add_argument('--language', type=str, default=None)
    parser.add_argument('--use_onnx', action='store_true', help='Use ONNX runtime for inference')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda, cpu, mps)')

    args = parser.parse_args()

    inferencer = WhisperInference(
        model_path=args.model_path,
        base_model=args.base_model,
        use_onnx=args.use_onnx,
        device=args.device
    )

    audio_path = Path(args.audio_path)

    if audio_path.is_file():
        audio_files = [str(audio_path)]
    elif audio_path.is_dir():
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        audio_files = [str(f) for f in audio_path.rglob('*') if f.suffix.lower() in audio_extensions]
    else:
        raise ValueError(f"Invalid audio path: {audio_path}")

    print(f"Found {len(audio_files)} audio file(s)")

    results = inferencer.transcribe_batch(audio_files, language=args.language)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"File: {result['audio_path']}\n")
            f.write(f"Transcription: {result['transcription']}\n")
            f.write(f"Processing Time: {result['processing_time']:.3f}s\n")
            f.write(f"Audio Duration: {result['audio_duration']:.3f}s\n")
            f.write(f"RTF: {result['rtf']:.3f}\n")
            f.write("-" * 80 + "\n")

            print(f"\nFile: {result['audio_path']}")
            print(f"Transcription: {result['transcription']}")
            print(f"RTF: {result['rtf']:.3f}")

    print(f"\nResults saved to {args.output_file}")

    avg_rtf = np.mean([r['rtf'] for r in results])
    print(f"Average RTF: {avg_rtf:.3f}")


if __name__ == '__main__':
    main()

import os
import argparse
import json
import torch
from tqdm import tqdm
from jiwer import wer, cer
import numpy as np
from pathlib import Path

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from utils import setup_logging, get_device
from dataset import create_dataloaders
from inference import WhisperInference


class ASREvaluator:
    def __init__(
        self,
        model_path: str,
        base_model: str = "openai/whisper-small",
        data_dir: str = "./data",
        batch_size: int = 4,
        use_synthetic: bool = False,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_synthetic = use_synthetic

        if device:
            self.device = torch.device(device)
        else:
            self.device = get_device()

        self.logger = setup_logging()

        self.inferencer = WhisperInference(
            model_path=model_path,
            base_model=base_model,
            device=str(self.device)
        )

        self._load_data()

    def _load_data(self):
        self.logger.info("Loading data loaders")

        _, _, self.test_loader = create_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            num_workers=2,
            sample_rate=16000,
            augmentation=None,
            processor=self.inferencer.processor,
            use_synthetic=self.use_synthetic
        )

        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")

    def evaluate(self) -> dict:
        self.logger.info("Starting evaluation...")

        references = []
        hypotheses = []
        rtf_scores = []

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            audio = batch['audio']
            transcripts = batch['transcript']

            for i in range(audio.shape[0]):
                audio_sample = audio[i]
                reference = transcripts[i]

                result = self.inferencer.transcribe(audio_sample)

                hypothesis = result['transcription']
                rtf = result['rtf']

                references.append(reference)
                hypotheses.append(hypothesis)
                rtf_scores.append(rtf)

        wer_score = wer(references, hypotheses)
        cer_score = cer(references, hypotheses)
        avg_rtf = np.mean(rtf_scores)

        results = {
            'wer': wer_score,
            'cer': cer_score,
            'avg_rtf': avg_rtf,
            'num_samples': len(references)
        }

        self.logger.info(f"\nEvaluation Results:")
        self.logger.info(f"==================")
        self.logger.info(f"WER: {wer_score * 100:.2f}%")
        self.logger.info(f"CER: {cer_score * 100:.2f}%")
        self.logger.info(f"Average RTF: {avg_rtf:.3f}")
        self.logger.info(f"Number of samples: {len(references)}")

        return results

    def evaluate_with_baseline(self, baseline_model: str = None) -> dict:
        if baseline_model is None:
            baseline_model = self.base_model

        self.logger.info(f"Evaluating baseline model: {baseline_model}")

        baseline_inferencer = WhisperInference(
            model_path=baseline_model,
            base_model=baseline_model,
            device=str(self.device)
        )

        references = []
        hypotheses_finetuned = []
        hypotheses_baseline = []

        for batch in tqdm(self.test_loader, desc="Evaluating with baseline"):
            audio = batch['audio']
            transcripts = batch['transcript']

            for i in range(audio.shape[0]):
                audio_sample = audio[i]
                reference = transcripts[i]

                result_finetuned = self.inferencer.transcribe(audio_sample)
                result_baseline = baseline_inferencer.transcribe(audio_sample)

                references.append(reference)
                hypotheses_finetuned.append(result_finetuned['transcription'])
                hypotheses_baseline.append(result_baseline['transcription'])

        wer_finetuned = wer(references, hypotheses_finetuned)
        wer_baseline = wer(references, hypotheses_baseline)
        wer_improvement = ((wer_baseline - wer_finetuned) / wer_baseline) * 100

        cer_finetuned = cer(references, hypotheses_finetuned)
        cer_baseline = cer(references, hypotheses_baseline)
        cer_improvement = ((cer_baseline - cer_finetuned) / cer_baseline) * 100

        results = {
            'wer_finetuned': wer_finetuned,
            'wer_baseline': wer_baseline,
            'wer_improvement': wer_improvement,
            'cer_finetuned': cer_finetuned,
            'cer_baseline': cer_baseline,
            'cer_improvement': cer_improvement,
            'num_samples': len(references)
        }

        self.logger.info(f"\nComparison Results:")
        self.logger.info(f"==================")
        self.logger.info(f"Fine-tuned WER: {wer_finetuned * 100:.2f}%")
        self.logger.info(f"Baseline WER: {wer_baseline * 100:.2f}%")
        self.logger.info(f"WER Improvement: {wer_improvement:.2f}%")
        self.logger.info(f"")
        self.logger.info(f"Fine-tuned CER: {cer_finetuned * 100:.2f}%")
        self.logger.info(f"Baseline CER: {cer_baseline * 100:.2f}%")
        self.logger.info(f"CER Improvement: {cer_improvement:.2f}%")

        return results

    def detailed_error_analysis(self, output_file: str = "error_analysis.json"):
        self.logger.info("Performing detailed error analysis...")

        error_samples = []

        for batch in tqdm(self.test_loader, desc="Error analysis"):
            audio = batch['audio']
            transcripts = batch['transcript']

            for i in range(audio.shape[0]):
                audio_sample = audio[i]
                reference = transcripts[i]

                result = self.inferencer.transcribe(audio_sample)
                hypothesis = result['transcription']

                sample_wer = wer(reference, hypothesis)
                sample_cer = cer(reference, hypothesis)

                error_samples.append({
                    'reference': reference,
                    'hypothesis': hypothesis,
                    'wer': sample_wer,
                    'cer': sample_cer,
                    'rtf': result['rtf']
                })

        error_samples.sort(key=lambda x: x['wer'], reverse=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(error_samples[:20], f, indent=2, ensure_ascii=False)

        self.logger.info(f"Top 20 errors saved to {output_file}")

        return error_samples


def main():
    parser = argparse.ArgumentParser(description='Evaluate ASR Model')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--base_model', type=str, default='openai/whisper-small')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_synthetic', action='store_true')
    parser.add_argument('--compare_baseline', action='store_true')
    parser.add_argument('--error_analysis', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = ASREvaluator(
        model_path=args.model_path,
        base_model=args.base_model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_synthetic=args.use_synthetic,
        device=args.device
    )

    if args.compare_baseline:
        results = evaluator.evaluate_with_baseline()
    else:
        results = evaluator.evaluate()

    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    if args.error_analysis:
        error_analysis_path = os.path.join(args.output_dir, 'error_analysis.json')
        evaluator.detailed_error_analysis(error_analysis_path)


if __name__ == '__main__':
    from typing import Optional
    main()

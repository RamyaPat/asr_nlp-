import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split


class AccentDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        augmentation=None,
        processor=None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.augmentation = augmentation
        self.processor = processor

        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        manifest_path = self.data_dir / f"{self.split}.json"

        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self._create_manifest_from_directory()

    def _create_manifest_from_directory(self) -> List[Dict]:
        audio_files = []

        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

        for ext in audio_extensions:
            audio_files.extend(list(self.data_dir.rglob(f"*{ext}")))

        data = []
        for audio_path in audio_files:
            transcript_path = audio_path.with_suffix('.txt')

            if transcript_path.exists():
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()

                data.append({
                    'audio_path': str(audio_path),
                    'transcript': transcript
                })

        train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }

        for split_name, split_data in splits.items():
            manifest_path = self.data_dir / f"{split_name}.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)

        return splits.get(self.split, data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        audio_path = item['audio_path']
        transcript = item['transcript']

        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        max_samples = int(self.max_duration * self.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        if self.split == 'train' and self.augmentation is not None:
            waveform = self.augmentation(waveform)

        result = {
            'audio': waveform.squeeze(0),
            'transcript': transcript,
            'audio_path': audio_path
        }

        if self.processor is not None:
            input_features = self.processor(
                waveform.squeeze(0).numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features.squeeze(0)

            result['input_features'] = input_features

        return result


class SyntheticAccentDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 100,
        sample_rate: int = 16000,
        duration: float = 3.0,
        augmentation=None,
        processor=None
    ):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.augmentation = augmentation
        self.processor = processor

        self.transcripts = self._generate_transcripts()

    def _generate_transcripts(self) -> List[str]:
        sample_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Hello world this is a test recording",
            "Speech recognition with accent adaptation",
            "Machine learning models for audio processing",
            "Natural language processing and understanding",
            "Automatic speech recognition systems",
            "Deep learning with neural networks",
            "Python programming for data science",
            "Artificial intelligence and robotics",
            "Computer vision and image processing"
        ]

        transcripts = []
        for i in range(self.num_samples):
            transcripts.append(sample_texts[i % len(sample_texts)])

        return transcripts

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        num_samples = int(self.duration * self.sample_rate)

        frequency = 440 + (idx % 10) * 50
        t = torch.linspace(0, self.duration, num_samples)
        waveform = torch.sin(2 * np.pi * frequency * t) * 0.3

        noise = torch.randn_like(waveform) * 0.05
        waveform = waveform + noise

        if self.augmentation is not None:
            waveform = self.augmentation(waveform.unsqueeze(0)).squeeze(0)

        transcript = self.transcripts[idx]

        result = {
            'audio': waveform,
            'transcript': transcript,
            'audio_path': f'synthetic_{idx}.wav'
        }

        if self.processor is not None:
            input_features = self.processor(
                waveform.numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features.squeeze(0)

            result['input_features'] = input_features

        return result


def collate_fn(batch: List[Dict]) -> Dict:
    audio_list = [item['audio'] for item in batch]
    transcripts = [item['transcript'] for item in batch]

    max_length = max(audio.shape[0] for audio in audio_list)

    padded_audio = []
    for audio in audio_list:
        if audio.shape[0] < max_length:
            padding = torch.zeros(max_length - audio.shape[0])
            audio = torch.cat([audio, padding])
        padded_audio.append(audio)

    result = {
        'audio': torch.stack(padded_audio),
        'transcript': transcripts,
    }

    if 'input_features' in batch[0]:
        input_features = [item['input_features'] for item in batch]
        result['input_features'] = torch.stack(input_features)

    return result


def create_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    sample_rate: int = 16000,
    augmentation=None,
    processor=None,
    use_synthetic: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if use_synthetic:
        train_dataset = SyntheticAccentDataset(
            num_samples=100,
            sample_rate=sample_rate,
            augmentation=augmentation,
            processor=processor
        )
        val_dataset = SyntheticAccentDataset(
            num_samples=20,
            sample_rate=sample_rate,
            augmentation=None,
            processor=processor
        )
        test_dataset = SyntheticAccentDataset(
            num_samples=20,
            sample_rate=sample_rate,
            augmentation=None,
            processor=processor
        )
    else:
        train_dataset = AccentDataset(
            data_dir=data_dir,
            split='train',
            sample_rate=sample_rate,
            augmentation=augmentation,
            processor=processor
        )
        val_dataset = AccentDataset(
            data_dir=data_dir,
            split='validation',
            sample_rate=sample_rate,
            augmentation=None,
            processor=processor
        )
        test_dataset = AccentDataset(
            data_dir=data_dir,
            split='test',
            sample_rate=sample_rate,
            augmentation=None,
            processor=processor
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

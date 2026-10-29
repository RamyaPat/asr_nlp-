import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple
import random


class AudioAugmentationPipeline:
    def __init__(
        self,
        sample_rate: int = 16000,
        speed_perturb_range: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        noise_snr_range: Tuple[float, float] = (5.0, 20.0),
        gain_range: Tuple[float, float] = (-6.0, 6.0),
        spec_augment_time_mask: int = 50,
        spec_augment_freq_mask: int = 10,
        apply_prob: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.speed_perturb_range = speed_perturb_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_snr_range = noise_snr_range
        self.gain_range = gain_range
        self.spec_augment_time_mask = spec_augment_time_mask
        self.spec_augment_freq_mask = spec_augment_freq_mask
        self.apply_prob = apply_prob

    def speed_perturbation(self, waveform: torch.Tensor, speed_factor: Optional[float] = None) -> torch.Tensor:
        if speed_factor is None:
            speed_factor = random.uniform(*self.speed_perturb_range)

        effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)]
        ]

        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects
        )
        return augmented

    def pitch_shift(self, waveform: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        if n_steps is None:
            n_steps = random.randint(*self.pitch_shift_range)

        effects = [
            ["pitch", str(n_steps * 100)],
            ["rate", str(self.sample_rate)]
        ]

        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects
        )
        return augmented

    def add_gaussian_noise(self, waveform: torch.Tensor, snr_db: Optional[float] = None) -> torch.Tensor:
        if snr_db is None:
            snr_db = random.uniform(*self.noise_snr_range)

        signal_power = waveform.norm(p=2)
        noise = torch.randn_like(waveform)

        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / torch.sqrt(torch.tensor(snr_linear))

        noise = noise / noise.norm(p=2) * noise_power

        return waveform + noise

    def gain_variation(self, waveform: torch.Tensor, gain_db: Optional[float] = None) -> torch.Tensor:
        if gain_db is None:
            gain_db = random.uniform(*self.gain_range)

        gain_linear = 10 ** (gain_db / 20.0)
        return waveform * gain_linear

    def add_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        reverberance = random.randint(0, 100)
        room_scale = random.randint(0, 100)
        stereo_depth = random.randint(0, 100)

        effects = [
            ["reverb", str(reverberance), str(100), str(room_scale), str(stereo_depth), str(0)],
        ]

        try:
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sample_rate, effects
            )
            return augmented
        except:
            return waveform

    def spec_augment(self, spectrogram: torch.Tensor) -> torch.Tensor:
        freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.spec_augment_freq_mask)
        time_masking = torchaudio.transforms.TimeMasking(time_mask_param=self.spec_augment_time_mask)

        augmented = freq_masking(spectrogram)
        augmented = time_masking(augmented)

        return augmented

    def __call__(self, waveform: torch.Tensor, apply_spec_augment: bool = False) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        augmented = waveform.clone()

        if random.random() < self.apply_prob:
            try:
                augmented = self.speed_perturbation(augmented)
            except:
                pass

        if random.random() < self.apply_prob:
            try:
                augmented = self.pitch_shift(augmented)
            except:
                pass

        if random.random() < self.apply_prob:
            augmented = self.add_gaussian_noise(augmented)

        if random.random() < self.apply_prob:
            augmented = self.gain_variation(augmented)

        if random.random() < self.apply_prob:
            try:
                augmented = self.add_reverb(augmented)
            except:
                pass

        if apply_spec_augment:
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate
            )(augmented)
            mel_spectrogram = self.spec_augment(mel_spectrogram)

        return augmented


class SimpleAugmentation:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def add_noise(self, waveform: torch.Tensor, snr_db: float = 15.0) -> torch.Tensor:
        signal_power = waveform.norm(p=2)
        noise = torch.randn_like(waveform)

        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / torch.sqrt(torch.tensor(snr_linear))

        noise = noise / noise.norm(p=2) * noise_power

        return waveform + noise

    def change_volume(self, waveform: torch.Tensor, gain_db: float = 3.0) -> torch.Tensor:
        gain_linear = 10 ** (gain_db / 20.0)
        return waveform * gain_linear

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            waveform = self.add_noise(waveform, snr_db=random.uniform(10, 20))
        if random.random() < 0.5:
            waveform = self.change_volume(waveform, gain_db=random.uniform(-3, 3))
        return waveform

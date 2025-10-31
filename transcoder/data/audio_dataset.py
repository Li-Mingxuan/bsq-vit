"""
Audio dataset loaders for speech and audio data.
Supports common audio datasets and mel-spectrogram extraction.
"""

import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from pathlib import Path


class AudioSpectrogramDataset(Dataset):
    """
    Generic audio dataset that loads audio files and converts them to mel-spectrograms.
    """
    def __init__(
        self,
        root,
        sample_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=160,
        target_length=1024,  # number of time frames
        audio_extensions=('.wav', '.flac', '.mp3', '.ogg'),
        normalize=True,
        augment=False,
    ):
        """
        Args:
            root: Root directory containing audio files
            sample_rate: Target sample rate for audio
            n_mels: Number of mel filterbanks
            n_fft: FFT size
            hop_length: Hop length for STFT
            target_length: Target length in frames (will pad/crop to this length)
            audio_extensions: Tuple of valid audio file extensions
            normalize: Whether to normalize spectrograms to dB scale
            augment: Whether to apply data augmentation
        """
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = target_length
        self.normalize = normalize
        self.augment = augment
        
        # Find all audio files
        self.audio_files = []
        for ext in audio_extensions:
            self.audio_files.extend(list(self.root.rglob(f'*{ext}')))
        
        if len(self.audio_files) == 0:
            raise RuntimeError(f"No audio files found in {root}")
        
        print(f"Found {len(self.audio_files)} audio files in {root}")
        
        # Setup mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
    
    def __len__(self):
        return len(self.audio_files)
    
    def _load_audio(self, path):
        """Load audio file and resample if necessary."""
        waveform, sr = torchaudio.load(path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform
    
    def _extract_spectrogram(self, waveform):
        """Extract mel-spectrogram from waveform."""
        # Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to dB scale
        if self.normalize:
            mel_spec = self.amplitude_to_db(mel_spec)
        else:
            # Just take log
            mel_spec = torch.log(mel_spec + 1e-9)
        
        return mel_spec
    
    def _process_length(self, mel_spec):
        """Pad or crop spectrogram to target length."""
        # mel_spec shape: (1, n_mels, time)
        current_length = mel_spec.shape[-1]
        
        if current_length < self.target_length:
            # Pad
            pad_length = self.target_length - current_length
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_length))
        elif current_length > self.target_length:
            # Crop randomly during training, from center during eval
            if self.augment:
                start = random.randint(0, current_length - self.target_length)
            else:
                start = (current_length - self.target_length) // 2
            mel_spec = mel_spec[..., start:start + self.target_length]
        
        return mel_spec
    
    def _augment(self, mel_spec):
        """Apply data augmentation to spectrogram."""
        # Time masking
        if random.random() < 0.5:
            time_mask_param = int(self.target_length * 0.1)
            time_masker = T.TimeMasking(time_mask_param=time_mask_param)
            mel_spec = time_masker(mel_spec)
        
        # Frequency masking
        if random.random() < 0.5:
            freq_mask_param = int(self.n_mels * 0.1)
            freq_masker = T.FrequencyMasking(freq_mask_param=freq_mask_param)
            mel_spec = freq_masker(mel_spec)
        
        return mel_spec
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio
        waveform = self._load_audio(audio_path)
        
        # Extract spectrogram
        mel_spec = self._extract_spectrogram(waveform)
        
        # Process length
        mel_spec = self._process_length(mel_spec)
        
        # Augment if training
        if self.augment:
            mel_spec = self._augment(mel_spec)
        
        # Squeeze channel dimension and transpose to (time, freq)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)  # (time, n_mels)
        
        # Return as tuple to match ImageFolder interface
        return mel_spec, 0  # dummy label


class LibriSpeechSpectrogramDataset(AudioSpectrogramDataset):
    """
    LibriSpeech dataset with mel-spectrogram extraction.
    Expects LibriSpeech directory structure.
    """
    def __init__(
        self,
        root,
        split='train-clean-100',
        **kwargs
    ):
        """
        Args:
            root: Root directory of LibriSpeech dataset
            split: Which split to use (e.g., 'train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean')
            **kwargs: Arguments passed to AudioSpectrogramDataset
        """
        split_path = os.path.join(root, split)
        if not os.path.exists(split_path):
            raise RuntimeError(f"LibriSpeech split {split} not found at {split_path}")
        
        super().__init__(split_path, **kwargs)
        print(f"Loaded LibriSpeech {split} split with {len(self)} samples")


class CommonVoiceSpectrogramDataset(AudioSpectrogramDataset):
    """
    Mozilla Common Voice dataset with mel-spectrogram extraction.
    """
    def __init__(
        self,
        root,
        split='train',
        **kwargs
    ):
        """
        Args:
            root: Root directory of Common Voice dataset
            split: Which split to use ('train', 'dev', 'test')
            **kwargs: Arguments passed to AudioSpectrogramDataset
        """
        # Common Voice typically has clips in a 'clips' subdirectory
        clips_path = os.path.join(root, 'clips')
        if not os.path.exists(clips_path):
            clips_path = root
        
        super().__init__(clips_path, **kwargs)
        print(f"Loaded Common Voice {split} split with {len(self)} samples")


class AudioFolderDataset(AudioSpectrogramDataset):
    """
    Simple audio folder dataset - loads all audio files from a directory.
    Similar to torchvision.datasets.ImageFolder but for audio.
    """
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
        print(f"Loaded AudioFolder dataset from {root} with {len(self)} samples")


def get_audio_dataset(dataset_type, root, **kwargs):
    """
    Factory function to get audio dataset.
    
    Args:
        dataset_type: Type of dataset ('audio_folder', 'librispeech', 'common_voice')
        root: Root directory of dataset
        **kwargs: Additional arguments for dataset
    
    Returns:
        Dataset instance
    """
    if dataset_type.lower() == 'audio_folder':
        return AudioFolderDataset(root, **kwargs)
    elif dataset_type.lower() == 'librispeech':
        return LibriSpeechSpectrogramDataset(root, **kwargs)
    elif dataset_type.lower() == 'common_voice':
        return CommonVoiceSpectrogramDataset(root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

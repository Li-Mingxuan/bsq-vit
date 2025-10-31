# Audio BSQ-ViT

This directory contains an adaptation of BSQ-ViT for audio/speech tokenization, inspired by AudioMAE and Conformer architectures.

## Overview

The audio version of BSQ-ViT processes audio spectrograms (mel-spectrograms) using a transformer-based architecture with optional Conformer-style convolution modules for better local feature extraction.

## Key Features

- **Audio Transformer Architecture**: Adapted from vision transformers to handle audio spectrograms
- **Conformer-style Convolution Modules**: Optional convolution modules for enhanced local feature modeling
- **Binary Spherical Quantization (BSQ)**: Efficient audio tokenization using binary quantization
- **Flexible Patch-based Processing**: Configurable time and frequency patch sizes
- **Multiple Dataset Support**: LibriSpeech, Common Voice, and generic audio folders

## Architecture Components

### AudioTransformerEncoder
- Processes mel-spectrograms using patch-based tokenization
- Supports Conformer-style convolution modules for local feature extraction
- Learnable positional embeddings for time-frequency patches

### AudioTransformerDecoder
- Reconstructs spectrograms from quantized representations
- Mirror architecture of the encoder

### AudioBSQModel
- Complete audio tokenization model
- Binary Spherical Quantization for efficient discrete representations
- Configurable quantization parameters

## Installation

Install additional audio dependencies:

```bash
mamba install -c conda-forge torchaudio librosa
```

## Data Preparation

### LibriSpeech
Download and extract LibriSpeech dataset:
```bash
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz
```

### Common Voice
Download from Mozilla Common Voice website and extract to your data directory.

### Custom Audio Dataset
Place your audio files (.wav, .flac, .mp3, .ogg) in a directory structure:
```
/path/to/audio/
├── train/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── val/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

## Training

### Basic Training
```bash
python scripts/main_audio_tokenizer.py \
    configs/tokenizer/audio_librispeech_bsq_b18.yaml \
    --output-dir ./outputs/audio_bsq \
    --use-ema
```

### Distributed Training
```bash
torchrun --nproc_per_node=8 scripts/main_audio_tokenizer.py \
    configs/tokenizer/audio_librispeech_bsq_b18.yaml \
    --output-dir ./outputs/audio_bsq \
    --use-ema
```

### Resume Training
```bash
python scripts/main_audio_tokenizer.py \
    configs/tokenizer/audio_librispeech_bsq_b18.yaml \
    --output-dir ./outputs/audio_bsq \
    --resume ./outputs/audio_bsq/checkpoint_0100000.pth
```

## Configuration

Key configuration parameters in the YAML files:

### Model Parameters
- `embed_dim`: Dimension of quantized embeddings (e.g., 18 for 18-bit codes)
- `input_size`: Number of mel-frequency bins (default: 128)
- `input_length`: Number of time frames (default: 1024)
- `patch_size`: Size of patches for both time and frequency dimensions
- `use_conv_module`: Enable Conformer-style convolution (recommended: True)
- `conv_kernel_size`: Kernel size for convolution module (default: 31)

### Audio Processing Parameters
- `sample_rate`: Audio sample rate (default: 16000 Hz)
- `n_mels`: Number of mel-frequency bins (default: 128)
- `n_fft`: FFT size (default: 1024)
- `hop_length`: Hop length for STFT (default: 160)
- `target_length`: Target length in frames (default: 1024)

### Training Parameters
- `batch_size`: Batch size per GPU
- `base_lr`: Base learning rate
- `max_iter`: Maximum training iterations
- `augment`: Enable data augmentation (time/frequency masking)

## Model Variants

### Standard Configuration (18-bit)
- `embed_dim: 18`
- `width: 768`
- `layers: 12`
- `heads: 12`

### Large Configuration (36-bit)
- `embed_dim: 36`
- `width: 1024`
- `layers: 24`
- `heads: 16`

### Small Configuration (9-bit)
- `embed_dim: 9`
- `width: 512`
- `layers: 6`
- `heads: 8`

## Evaluation

The model outputs:
- **MSE**: Mean Squared Error on spectrograms
- **MAE**: Mean Absolute Error on spectrograms
- **Reconstruction Loss**: Overall reconstruction quality

## Audio-Specific Design Choices

### Differences from Image BSQ-ViT

1. **Input Format**: Mel-spectrograms (time × frequency) instead of images (height × width)
2. **Patch Strategy**: Separate time and frequency patch sizes for better flexibility
3. **Convolution Modules**: Optional Conformer-style convolutions for local temporal modeling
4. **Normalization**: dB-scale spectrograms (typically -80 to 0 dB range)
5. **Data Augmentation**: Time and frequency masking (SpecAugment-style)
6. **No Perceptual Loss**: Removed LPIPS loss (designed for images)

### Inspiration from AudioMAE and Conformer

1. **AudioMAE**: 
   - Patch-based processing of spectrograms
   - Positional embeddings for time-frequency patches
   - Masked autoencoding capability (can be added)

2. **Conformer**:
   - Convolution modules for local feature extraction
   - GLU activation in convolution layers
   - Batch normalization in convolution paths

## Example Workflow

```python
import torch
from transcoder.models.audiobsq import AudioBSQModel

# Create model
config = {
    'input_size': 128,
    'input_length': 1024,
    'patch_size': 16,
    'width': 768,
    'layers': 12,
    'heads': 12,
    'use_conv_module': True,
}

model = AudioBSQModel(
    audioconfig=config,
    embed_dim=18,
    l2_norm=True,
)

# Process audio (mel-spectrogram)
audio = torch.randn(4, 1024, 128)  # (batch, time, freq)
reconstructed, quant_loss, info = model(audio)

# Access quantized codes
encoded, _, _ = model.encode(audio)
print(f"Encoded shape: {encoded.shape}")
```

## Citation

If you use this audio adaptation, please cite both the original BSQ-ViT paper and the inspirational works:

```bibtex
@article{zhao2024bsqvit,
  title={Image and Video Tokenization with Binary Spherical Quantization},
  author={Zhao, Yue and Xiong, Yuanjun, and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv preprint arXiv:2406.07548},
  year={2024}
}

@article{huang2022audiomae,
  title={Masked autoencoders that listen},
  author={Huang, Po-Yao and Xu, Hu and Li, Juncheng and others},
  journal={NeurIPS},
  year={2022}
}

@article{gulati2020conformer,
  title={Conformer: Convolution-augmented transformer for speech recognition},
  author={Gulati, Anmol and Qin, James and others},
  journal={INTERSPEECH},
  year={2020}
}
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `input_length` (number of time frames)
- Enable gradient checkpointing: `grad_checkpointing: True` in config

### Poor Reconstruction Quality
- Increase model capacity (`width`, `layers`)
- Increase `embed_dim` (more quantization bits)
- Enable Conformer convolution modules
- Adjust `clamp_range` to match your audio data

### Training Instability
- Reduce learning rate
- Increase warm-up steps
- Check audio normalization (should be in dB scale)
- Enable gradient clipping (already enabled by default)

## Future Improvements

- [ ] Masked autoencoding pre-training (AudioMAE-style)
- [ ] Multi-scale spectrogram processing
- [ ] Vocoder integration for audio synthesis
- [ ] Additional audio datasets (Speech Commands, AudioSet)
- [ ] Audio-specific evaluation metrics (SI-SNR, PESQ, STOI)
- [ ] Streaming audio processing support

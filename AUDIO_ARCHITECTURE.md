# Audio BSQ-ViT Architecture Details

This document provides detailed technical information about the audio adaptation of BSQ-ViT.

## Architecture Overview

```
Audio Input (Mel-Spectrogram)
    ↓
[Patch Embedding]
    ↓
[Positional Embedding]
    ↓
[Audio Transformer Encoder]
    - Multi-head Self-Attention
    - Conformer Convolution Module (optional)
    - Feed-Forward Network
    ↓
[Quantization Embedding Layer]
    ↓
[Binary Spherical Quantizer (BSQ)]
    ↓
[Post-Quantization Embedding Layer]
    ↓
[Audio Transformer Decoder]
    - Multi-head Self-Attention
    - Conformer Convolution Module (optional)
    - Feed-Forward Network
    ↓
[Reconstruction Head]
    ↓
Reconstructed Spectrogram
```

## Key Components

### 1. Patch Embedding

**Purpose**: Convert 2D audio spectrogram into sequence of patch embeddings.

**Image BSQ-ViT**:
```python
# Patchify: (B, C, H, W) -> (B, num_patches, patch_dim)
# C=3 (RGB), H=256, W=256, patch_size=8
# num_patches = (256/8) * (256/8) = 1024
```

**Audio BSQ-ViT**:
```python
# Patchify: (B, T, F) -> (B, num_patches, patch_dim)
# T=1024 (time frames), F=128 (mel bins), patch_size=16
# num_patches = (1024/16) * (128/16) = 512
```

**Key Differences**:
- Audio uses 1-channel spectrograms vs 3-channel RGB images
- Time-frequency patches vs spatial patches
- Separate time/frequency patch sizes possible

### 2. Positional Embeddings

**Image BSQ-ViT**:
```python
# 2D spatial positional embedding
# Shape: (H/patch_size * W/patch_size, hidden_dim)
self.positional_embedding = nn.Parameter(
    scale * torch.randn(grid_h * grid_w, width)
)
```

**Audio BSQ-ViT**:
```python
# 2D time-frequency positional embedding
# Shape: (T/patch_size * F/patch_size, hidden_dim)
self.positional_embedding = nn.Parameter(
    scale * torch.randn(num_patches, width)
)
```

**Key Differences**:
- Similar structure but different semantics
- Time axis is temporal, frequency axis is spectral
- Could potentially use 1D temporal embedding + learned frequency embedding

### 3. Conformer Convolution Module

**New Addition (inspired by Conformer)**:

```python
class ConvolutionModule(nn.Module):
    """
    Pointwise Conv -> GLU -> Depthwise Conv -> BatchNorm -> Swish -> Pointwise Conv
    """
    def __init__(self, d_model, kernel_size=31):
        self.pointwise_conv1 = nn.Conv1d(d_model, 2*d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
```

**Why Conformer for Audio?**:
- Captures local temporal patterns in audio
- Combines CNN's local feature extraction with Transformer's global modeling
- Proven effective for speech recognition (Conformer paper)
- Depthwise convolution is parameter-efficient

### 4. Attention Mechanism

**Both models use standard multi-head self-attention**:
```python
self.attn = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=n_head,
    dropout=attn_drop
)
```

**Key Parameters**:
- Image: 768 hidden dim, 12 heads (64 dim per head)
- Audio: 768 hidden dim, 12 heads (64 dim per head)
- Same architecture, different data modality

### 5. Binary Spherical Quantizer (BSQ)

**Identical between image and audio**:
```python
class BinarySphericalQuantizer:
    """
    Maps continuous vectors to binary codes on a hypersphere
    - L2 normalize input vectors
    - Quantize each dimension to {-1, +1}
    - Result: compact binary codes
    """
```

**Why BSQ works for audio**:
- Audio spectrograms have similar statistical properties to images
- Both are 2D structured data with local correlations
- Efficient compression through binary codes
- Perceptual quality maintained through spherical constraint

### 6. Decoder Architecture

**Image BSQ-ViT**:
```python
# Reconstruct RGB image
output_channels = 3  # or 6 with logit_laplace
output_dim = 3 * patch_h * patch_w
```

**Audio BSQ-ViT**:
```python
# Reconstruct mel-spectrogram
output_channels = 1  # single channel spectrogram
output_dim = time_patch * freq_patch
```

## Architecture Comparison Table

| Component | Image BSQ-ViT | Audio BSQ-ViT | Notes |
|-----------|---------------|---------------|-------|
| Input Shape | (B, 3, 256, 256) | (B, 1024, 128) | RGB vs Mel-spec |
| Input Type | RGB Image | Mel-Spectrogram | Visual vs Acoustic |
| Patch Size | 8×8 or 16×16 | 16×16 (T×F) | Spatial vs Time-Freq |
| Num Patches | 1024 (256/8)² | 512 (64×8) | Depends on input size |
| Positional Emb | 2D Spatial | 2D Time-Freq | Same structure |
| Convolution | ✗ | ✓ (Optional) | Conformer-style |
| Attention | Multi-head | Multi-head | Same |
| Quantization | BSQ | BSQ | Identical |
| Output Range | [0,1] or [-1,1] | [-80,0] dB | Different normalization |
| Loss Function | MSE + LPIPS | MSE only | No perceptual loss for audio |

## Conformer Integration

The Conformer-style convolution module is integrated into each transformer block:

```
Input
  ↓
LayerNorm → Multi-Head Attention → Add & Scale
  ↓                                    ↓
  └────────────────────────────────────┘
  ↓
LayerNorm → Convolution Module → Add & Scale  [NEW]
  ↓                                  ↓
  └──────────────────────────────────┘
  ↓
LayerNorm → Feed-Forward Network → Add & Scale
  ↓                                   ↓
  └───────────────────────────────────┘
  ↓
Output
```

**Benefits**:
1. Local temporal modeling via depthwise convolution
2. Efficient parameter usage (depthwise separable)
3. Complementary to global attention
4. Proven in Conformer for ASR

## Data Processing Pipeline

### Image BSQ-ViT
```
Raw Image
  ↓ [Load & Decode]
RGB Image (H×W×3)
  ↓ [Resize & Normalize]
Tensor (3×256×256) in [0,1] or [-1,1]
  ↓ [Model]
Reconstructed Image
```

### Audio BSQ-ViT
```
Raw Audio Waveform
  ↓ [Load & Resample to 16kHz]
Mono Waveform (T×1)
  ↓ [STFT & Mel-Filterbank]
Mel-Spectrogram (Time×Freq)
  ↓ [Log-scale & Normalize]
Tensor (1024×128) in dB [-80, 0]
  ↓ [Model]
Reconstructed Mel-Spectrogram
  ↓ [Optional: Griffin-Lim or Vocoder]
Reconstructed Waveform
```

## Training Differences

| Aspect | Image | Audio |
|--------|-------|-------|
| Batch Size | 32 | 32 |
| Learning Rate | 4e-7 | 4e-7 |
| Input Size | 256×256×3 | 1024×128×1 |
| Augmentation | RandomCrop, Flip | SpecAugment (Time/Freq mask) |
| Perceptual Loss | LPIPS (VGG) | None |
| Discriminator | StyleGAN | PatchGAN |
| Evaluation | FID, PSNR, SSIM | MSE, MAE |

## Design Rationale

### Why These Adaptations?

1. **Conformer Convolution**:
   - Audio has strong local temporal structure
   - Speech phonemes span ~10-100ms windows
   - Convolution captures these local patterns efficiently

2. **No Perceptual Loss**:
   - LPIPS is designed for images (VGG features)
   - Audio perception is different (frequency content, phase)
   - Future work: audio-specific perceptual losses

3. **Different Normalization**:
   - Images: [0,1] or [-1,1]
   - Audio: [-80, 0] dB scale
   - Matches typical audio processing conventions

4. **Patch Size Choice**:
   - 16×16 patches balance:
     - Temporal resolution (~100ms at 16kHz with 160 hop)
     - Frequency resolution (~8 mel bins per patch)
     - Computational efficiency

## AudioMAE Inspiration

Key ideas from AudioMAE incorporated:

1. **Patch-based Processing**: Treat audio as sequence of patches
2. **Positional Embeddings**: Learn time-frequency structure
3. **Masked Modeling** (can be added): Pre-train with masked patches
4. **Spectrogram Input**: Use mel-spectrograms, not raw waveforms

## Future Enhancements

Potential improvements inspired by recent audio research:

1. **Masked Pre-training**:
   ```python
   # Mask random patches during pre-training (AudioMAE-style)
   mask_ratio = 0.75
   masked_patches = random_mask(patches, mask_ratio)
   ```

2. **Multi-scale Processing**:
   ```python
   # Process multiple time/frequency resolutions
   scales = [16, 32, 64]  # Different patch sizes
   ```

3. **Audio-specific Perceptual Loss**:
   ```python
   # Use pre-trained audio networks (Wav2Vec2, HuBERT)
   perceptual_loss = wav2vec_features(recon, target)
   ```

4. **Vocoder Integration**:
   ```python
   # Direct waveform generation
   waveform = vocoder(mel_spectrogram)
   ```

5. **Streaming Support**:
   ```python
   # Process audio in chunks for real-time applications
   chunk_size = 256  # frames
   ```

## References

1. **BSQ-ViT**: Zhao et al., "Image and Video Tokenization with Binary Spherical Quantization", 2024
2. **AudioMAE**: Huang et al., "Masked Autoencoders that Listen", NeurIPS 2022
3. **Conformer**: Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition", INTERSPEECH 2020
4. **SpecAugment**: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR", INTERSPEECH 2019

## Implementation Notes

### Memory Optimization
```python
# Enable gradient checkpointing for large models
model = AudioBSQModel(..., grad_checkpointing=True)
```

### Mixed Precision Training
```python
# Use bfloat16 for better stability
scaler = torch.cuda.amp.GradScaler(enabled=use_bf16)
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
```

### Distributed Training
```python
# Multi-GPU training with DDP
model = torch.nn.parallel.DistributedDataParallel(model)
```

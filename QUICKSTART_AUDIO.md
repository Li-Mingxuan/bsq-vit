# Quick Start Guide for Audio BSQ-ViT

This guide will help you get started with audio tokenization using BSQ-ViT in just a few minutes.

## Prerequisites

Make sure you have the BSQ-ViT environment set up:

```bash
# Create and activate the environment
mamba env create -f bsqvit-env.yaml
mamba activate bsqvit

# Install audio dependencies
mamba install -c conda-forge torchaudio librosa
```

## Step 1: Test the Installation

Run the test script to verify everything is working:

```bash
python scripts/test_audio_model.py
```

You should see output like:
```
================================================================================
Testing Audio BSQ Model
================================================================================
1. Creating Audio BSQ Model...
   Total parameters: 108,862,464
...
✓ All tests passed!
```

## Step 2: Run the Example

Try the interactive example:

```bash
python examples/audio_example.py
```

This will:
- Create an audio BSQ model
- Generate a dummy spectrogram
- Encode and decode it
- Show reconstruction quality

## Step 3: Prepare Your Audio Data

### Option A: Use LibriSpeech (Recommended for beginners)

```bash
# Download LibriSpeech train-clean-100 (6.3GB)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# Download dev-clean for validation (337MB)
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz

# Update paths in config
# Edit configs/tokenizer/audio_librispeech_bsq_b18.yaml
# Change: root: '/path/to/LibriSpeech'
# To: root: './LibriSpeech'  (or your actual path)
```

### Option B: Use Your Own Audio Files

```bash
# Organize your audio files like this:
mkdir -p my_audio_data/train
mkdir -p my_audio_data/val

# Put your training audio files in train/
# Put your validation audio files in val/

# Update config to use audio_simple_bsq_b18.yaml
# And change the paths to point to your data
```

## Step 4: Train a Small Model

### Single GPU Training

```bash
python scripts/main_audio_tokenizer.py \
    configs/tokenizer/audio_simple_bsq_b18.yaml \
    --output-dir ./outputs/my_first_audio_model \
    --use-ema \
    --eval-freq 5000 \
    --save-freq 1000
```

### Multi-GPU Training (4 GPUs)

```bash
torchrun --nproc_per_node=4 scripts/main_audio_tokenizer.py \
    configs/tokenizer/audio_librispeech_bsq_b18.yaml \
    --output-dir ./outputs/audio_bsq_distributed \
    --use-ema
```

## Step 5: Monitor Training

### Using WandB (Recommended)

If you have wandb configured, you'll see:
- Training/validation loss curves
- Learning rate schedules
- Reconstruction quality metrics

Access at: https://wandb.ai/your-project/audio-transcoder

### Using Local Logs

Training progress is printed to console:
```
Iter [100/500000] train/loss: 0.1234 train/quant_loss: 0.0567 ...
```

## Step 6: Evaluate Your Model

```bash
python scripts/main_audio_tokenizer.py \
    configs/tokenizer/audio_simple_bsq_b18.yaml \
    --output-dir ./outputs/my_first_audio_model \
    --resume ./outputs/my_first_audio_model/checkpoint_0010000.pth \
    --evaluate
```

## Step 7: Use Your Trained Model

```python
import torch
from transcoder.models.audiobsq import AudioBSQModel
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load('configs/tokenizer/audio_simple_bsq_b18.yaml')

# Create model
model = AudioBSQModel(**config.model.params)

# Load checkpoint
checkpoint = torch.load('outputs/my_first_audio_model/checkpoint_0010000.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Use model for encoding
audio = torch.randn(1, 1024, 128)  # Your mel-spectrogram
with torch.no_grad():
    encoded, _, _ = model.encode(audio)
    reconstructed = model.decode(encoded)
```

## Common Issues and Solutions

### Out of Memory

**Problem**: CUDA out of memory error

**Solution**:
```yaml
# In your config file, reduce:
data:
  batch_size: 16  # Reduce from 32
  
model:
  params:
    audioconfig:
      width: 512  # Reduce from 768
      layers: 6   # Reduce from 12
```

### Slow Training

**Problem**: Training is very slow

**Solution**:
```bash
# Enable gradient checkpointing
# Edit config:
model:
  grad_checkpointing: True
  
# Or use smaller model (see above)

# Or reduce input length:
data:
  target_length: 512  # Reduce from 1024
```

### No Audio Files Found

**Problem**: "No audio files found" error

**Solution**:
```bash
# Check your directory structure
ls -R /path/to/audio/data

# Make sure you have .wav, .flac, .mp3, or .ogg files
# Check the path in your config file matches
```

### Poor Reconstruction Quality

**Problem**: Reconstructed audio sounds bad

**Solution**:
```yaml
# 1. Train longer (increase max_iter)
# 2. Use larger model (increase embed_dim, width, layers)
# 3. Enable Conformer convolutions:
model:
  params:
    audioconfig:
      use_conv_module: True
      
# 4. Adjust learning rate:
optimizer:
  base_lr: 2e-7  # Try lower learning rate
```

## Configuration Tips

### For Speech (LibriSpeech, Common Voice)

```yaml
data:
  sample_rate: 16000
  n_mels: 80  # Fewer mel bins for speech
  target_length: 1024
  
model:
  params:
    audioconfig:
      input_size: 80
      use_conv_module: True  # Recommended for speech
```

### For Music

```yaml
data:
  sample_rate: 22050  # Higher sample rate
  n_mels: 128
  n_fft: 2048  # Larger FFT
  target_length: 2048  # Longer context
  
model:
  params:
    audioconfig:
      input_size: 128
      input_length: 2048
      use_conv_module: True
```

### For Environmental Sounds

```yaml
data:
  sample_rate: 16000
  n_mels: 128
  target_length: 512  # Shorter context
  
model:
  params:
    audioconfig:
      input_size: 128
      input_length: 512
```

## Next Steps

Once you have a working model:

1. **Fine-tune on your specific domain**: Use your trained model as initialization
2. **Experiment with architecture**: Try different layer counts, hidden dimensions
3. **Try different quantization bits**: Change `embed_dim` (9, 18, 36 bits)
4. **Add vocoder**: Convert spectrograms back to audio waveforms
5. **Explore downstream tasks**: Use encoded representations for classification, generation

## Getting Help

- Check [AUDIO_README.md](AUDIO_README.md) for detailed documentation
- See [AUDIO_ARCHITECTURE.md](AUDIO_ARCHITECTURE.md) for architecture details
- Open an issue on GitHub for bugs or questions

## Resources

- **Paper**: [BSQ-ViT arXiv](http://arxiv.org/abs/2406.07548)
- **AudioMAE**: [Paper](https://arxiv.org/abs/2207.06405)
- **Conformer**: [Paper](https://arxiv.org/abs/2005.08100)
- **LibriSpeech**: [OpenSLR](https://www.openslr.org/12/)

Happy audio tokenizing! 🎵

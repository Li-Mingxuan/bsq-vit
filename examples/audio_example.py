"""
Example script demonstrating how to use the Audio BSQ model.

This script shows:
1. How to create an audio BSQ model
2. How to encode audio spectrograms
3. How to decode from quantized representations
4. How to save and load models
"""

import torch
import torch.nn as nn
from transcoder.models.audiobsq import AudioBSQModel


def create_audio_bsq_model(
    embed_dim=18,
    input_size=128,
    input_length=1024,
    patch_size=16,
    width=768,
    layers=12,
    heads=12,
    use_conformer=True,
):
    """
    Create an Audio BSQ model with specified configuration.
    
    Args:
        embed_dim: Number of bits for quantization (e.g., 18 for 18-bit)
        input_size: Number of mel-frequency bins
        input_length: Number of time frames
        patch_size: Size of patches (both time and frequency)
        width: Hidden dimension of transformer
        layers: Number of transformer layers
        heads: Number of attention heads
        use_conformer: Whether to use Conformer-style convolution modules
    
    Returns:
        Audio BSQ model instance
    """
    audioconfig = {
        'input_size': input_size,
        'input_length': input_length,
        'patch_size': patch_size,
        'width': width,
        'layers': layers,
        'heads': heads,
        'mlp_ratio': 4.0,
        'drop_rate': 0.0,
        'use_conv_module': use_conformer,
        'conv_kernel_size': 31,
    }
    
    model = AudioBSQModel(
        audioconfig=audioconfig,
        embed_dim=embed_dim,
        embed_group_size=1,
        l2_norm=True,
        post_q_l2_norm=True,
        clamp_range=(-80, 0),  # dB range for spectrograms
    )
    
    return model


def encode_audio(model, audio_spectrogram):
    """
    Encode audio spectrogram to quantized representation.
    
    Args:
        model: Audio BSQ model
        audio_spectrogram: Audio mel-spectrogram (batch, time, freq)
    
    Returns:
        Quantized representation, loss, and info dict
    """
    model.eval()
    with torch.no_grad():
        encoded, quant_loss, info = model.encode(audio_spectrogram)
    return encoded, quant_loss, info


def decode_audio(model, encoded):
    """
    Decode quantized representation back to audio spectrogram.
    
    Args:
        model: Audio BSQ model
        encoded: Quantized representation
    
    Returns:
        Reconstructed audio spectrogram (batch, time, freq)
    """
    model.eval()
    with torch.no_grad():
        decoded = model.decode(encoded)
    return decoded


def full_reconstruction(model, audio_spectrogram):
    """
    Full encode-decode reconstruction.
    
    Args:
        model: Audio BSQ model
        audio_spectrogram: Audio mel-spectrogram (batch, time, freq)
    
    Returns:
        Reconstructed audio spectrogram
    """
    model.eval()
    with torch.no_grad():
        reconstructed, quant_loss, info = model(audio_spectrogram)
    return reconstructed, quant_loss, info


def save_model(model, path):
    """Save model checkpoint."""
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'embed_dim': model.embed_dim,
            # Add other relevant config parameters
        }
    }, path)
    print(f"Model saved to {path}")


def load_model(path, model):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model loaded from {path}")
    return model


def main():
    """Main example demonstrating audio BSQ usage."""
    print("=" * 80)
    print("Audio BSQ Model Example")
    print("=" * 80)
    
    # 1. Create model
    print("\n1. Creating Audio BSQ Model...")
    model = create_audio_bsq_model(
        embed_dim=18,
        input_size=128,
        input_length=1024,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        use_conformer=True,
    )
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 2. Create dummy audio spectrogram
    print("\n2. Creating dummy audio spectrogram...")
    batch_size = 4
    audio = torch.randn(batch_size, 1024, 128)  # (batch, time, freq)
    print(f"   Audio shape: {audio.shape}")
    
    # 3. Encode audio
    print("\n3. Encoding audio...")
    encoded, quant_loss, info = encode_audio(model, audio)
    print(f"   Encoded shape: {encoded.shape}")
    print(f"   Quantization loss: {quant_loss.item():.6f}")
    
    # 4. Decode audio
    print("\n4. Decoding audio...")
    decoded = decode_audio(model, encoded)
    print(f"   Decoded shape: {decoded.shape}")
    
    # 5. Full reconstruction
    print("\n5. Full reconstruction...")
    reconstructed, quant_loss, info = full_reconstruction(model, audio)
    print(f"   Reconstructed shape: {reconstructed.shape}")
    
    # 6. Compute reconstruction quality
    print("\n6. Computing reconstruction quality...")
    mse = torch.nn.functional.mse_loss(reconstructed, audio)
    mae = torch.nn.functional.l1_loss(reconstructed, audio)
    print(f"   MSE: {mse.item():.6f}")
    print(f"   MAE: {mae.item():.6f}")
    
    # 7. Save model (commented out by default)
    # print("\n7. Saving model...")
    # save_model(model, 'audio_bsq_checkpoint.pth')
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()

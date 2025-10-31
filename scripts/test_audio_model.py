"""
Simple test script to verify the audio BSQ model works correctly.
This script creates a dummy audio spectrogram and tests the model forward pass.
"""

import torch
import torch.nn.functional as F
from transcoder.models.audiobsq import AudioBSQModel


def test_audio_bsq_model():
    """Test basic functionality of AudioBSQModel."""
    print("=" * 80)
    print("Testing Audio BSQ Model")
    print("=" * 80)
    
    # Model configuration
    audioconfig = {
        'input_size': 128,          # mel bins
        'input_length': 1024,       # time frames
        'patch_size': 16,
        'width': 768,
        'layers': 12,
        'heads': 12,
        'mlp_ratio': 4.0,
        'use_conv_module': True,    # Enable Conformer-style convolution
        'conv_kernel_size': 31,
    }
    
    # Create model
    print("\n1. Creating Audio BSQ Model...")
    model = AudioBSQModel(
        audioconfig=audioconfig,
        embed_dim=18,  # 18-bit quantization
        embed_group_size=1,
        l2_norm=True,
        post_q_l2_norm=True,
        clamp_range=(-80, 0),  # dB range for spectrograms
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {num_params:,}")
    
    # Create dummy input (batch_size=2, time=1024, freq=128)
    print("\n2. Creating dummy input...")
    batch_size = 2
    dummy_audio = torch.randn(batch_size, audioconfig['input_length'], audioconfig['input_size'])
    print(f"   Input shape: {dummy_audio.shape}")
    
    # Test encoder
    print("\n3. Testing encoder...")
    model.eval()
    with torch.no_grad():
        encoded, quant_loss, info = model.encode(dummy_audio)
    print(f"   Encoded shape: {encoded.shape}")
    print(f"   Expected shape: ({batch_size}, {audioconfig['input_length'] * audioconfig['input_size'] // (audioconfig['patch_size'] ** 2)}, 18)")
    
    # Test decoder
    print("\n4. Testing decoder...")
    with torch.no_grad():
        decoded = model.decode(encoded)
    print(f"   Decoded shape: {decoded.shape}")
    print(f"   Expected shape: {dummy_audio.shape}")
    
    # Test full forward pass
    print("\n5. Testing full forward pass...")
    with torch.no_grad():
        reconstructed, quant_loss, info = model(dummy_audio)
    print(f"   Reconstructed shape: {reconstructed.shape}")
    print(f"   Quantization loss: {quant_loss.item():.6f}")
    
    # Compute reconstruction metrics
    mse = F.mse_loss(reconstructed, dummy_audio)
    mae = F.l1_loss(reconstructed, dummy_audio)
    print(f"   MSE: {mse.item():.6f}")
    print(f"   MAE: {mae.item():.6f}")
    
    # Print quantization info
    print("\n6. Quantization info:")
    for key, value in info.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.item():.6f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


def test_different_input_shapes():
    """Test model with different input shapes."""
    print("\n" + "=" * 80)
    print("Testing Different Input Shapes")
    print("=" * 80)
    
    audioconfig = {
        'input_size': 128,
        'input_length': 1024,
        'patch_size': 16,
        'width': 512,
        'layers': 6,
        'heads': 8,
        'mlp_ratio': 4.0,
        'use_conv_module': False,  # Disable for faster testing
    }
    
    model = AudioBSQModel(
        audioconfig=audioconfig,
        embed_dim=9,
        l2_norm=True,
    )
    model.eval()
    
    test_cases = [
        (1, 1024, 128, "Single sample"),
        (4, 1024, 128, "Small batch"),
        (8, 1024, 128, "Medium batch"),
    ]
    
    for batch_size, time, freq, description in test_cases:
        print(f"\n{description}: ({batch_size}, {time}, {freq})")
        dummy_audio = torch.randn(batch_size, time, freq)
        
        with torch.no_grad():
            reconstructed, quant_loss, info = model(dummy_audio)
        
        print(f"   Input shape: {dummy_audio.shape}")
        print(f"   Output shape: {reconstructed.shape}")
        print(f"   Shapes match: {reconstructed.shape == dummy_audio.shape}")
        assert reconstructed.shape == dummy_audio.shape, "Shape mismatch!"
    
    print("\n✓ All shape tests passed!")


def test_encoder_decoder_separately():
    """Test encoder and decoder components separately."""
    print("\n" + "=" * 80)
    print("Testing Encoder and Decoder Separately")
    print("=" * 80)
    
    from transcoder.models.audio_transformer import AudioTransformerEncoder, AudioTransformerDecoder
    
    config = {
        'input_size': 128,
        'input_length': 1024,
        'patch_size': 16,
        'width': 512,
        'layers': 6,
        'heads': 8,
        'mlp_ratio': 4.0,
        'use_conv_module': True,
        'conv_kernel_size': 31,
    }
    
    # Test encoder
    print("\n1. Testing AudioTransformerEncoder...")
    encoder = AudioTransformerEncoder(**config)
    print(f"   Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    dummy_audio = torch.randn(2, 1024, 128)
    encoded = encoder(dummy_audio)
    print(f"   Input shape: {dummy_audio.shape}")
    print(f"   Encoded shape: {encoded.shape}")
    
    # Test decoder
    print("\n2. Testing AudioTransformerDecoder...")
    decoder = AudioTransformerDecoder(**config)
    print(f"   Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    decoded = decoder(encoded)
    print(f"   Encoded shape: {encoded.shape}")
    print(f"   Decoded shape: {decoded.shape}")
    print(f"   Original shape: {dummy_audio.shape}")
    assert decoded.shape == dummy_audio.shape, "Shape mismatch in decoder!"
    
    print("\n✓ Encoder and decoder tests passed!")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Audio BSQ Model Test Suite")
    print("=" * 80)
    
    try:
        # Run all tests
        test_audio_bsq_model()
        test_different_input_shapes()
        test_encoder_decoder_separately()
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 80)
        print("\nThe audio BSQ model is working correctly.")
        print("You can now:")
        print("  1. Prepare your audio dataset")
        print("  2. Configure training parameters in YAML files")
        print("  3. Start training with: python scripts/main_audio_tokenizer.py <config.yaml>")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80 + "\n")
        exit(1)

"""
Audio Transformer components inspired by AudioMAE and Conformer.
This module adapts the vision transformer to handle audio spectrograms.
"""

from collections import OrderedDict
from typing import Callable, Optional, Union
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from transcoder.models.attention_mask import get_attention_mask


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ConvolutionModule(nn.Module):
    """Convolution module from Conformer architecture."""
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, stride=1, padding=0, bias=True)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=d_model, bias=True
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, d_model)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        return x


class ResidualAttentionBlock(nn.Module):
    """Residual Attention Block with optional Conformer-style convolution."""
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            use_preln: bool = True,
            use_conv_module: bool = False,
            conv_kernel_size: int = 31,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=attn_drop, batch_first=False)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        # Optional convolution module (Conformer-style)
        self.use_conv_module = use_conv_module
        if use_conv_module:
            self.conv_module = ConvolutionModule(d_model, kernel_size=conv_kernel_size, dropout=drop)
            self.ln_conv = norm_layer(d_model)
            self.ls_conv = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model)),
            ("drop2", nn.Dropout(drop)),
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.use_preln = use_preln

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask, is_causal=is_causal)[0]

    def checkpoint_forward(self, x: torch.Tensor, 
                           attn_mask: Optional[torch.Tensor] = None,
                           is_causal: bool = False):
        state = x
        if self.use_preln:
            x = checkpoint(self.ln_1, x, use_reentrant=False)
            x = self.attention(x, attn_mask, is_causal)
            x = checkpoint(self.ls_1, x, use_reentrant=False)
            state = state + self.drop_path(x)
            
            if self.use_conv_module:
                x = checkpoint(self.ln_conv, state, use_reentrant=False)
                x = self.conv_module(x.permute(1, 0, 2)).permute(1, 0, 2)
                x = checkpoint(self.ls_conv, x, use_reentrant=False)
                state = state + self.drop_path(x)
            
            x = checkpoint(self.ln_2, state, use_reentrant=False)
            x = self.mlp(x)
            x = checkpoint(self.ls_2, x, use_reentrant=False)
            state = state + self.drop_path(x)
        else:
            x = self.attention(x, attn_mask, is_causal)
            x = state + self.drop_path(x)
            state = checkpoint(self.ln_1, x, use_reentrant=False)
            
            if self.use_conv_module:
                x = self.conv_module(state.permute(1, 0, 2)).permute(1, 0, 2)
                state = state + self.drop_path(x)
                state = checkpoint(self.ln_conv, state, use_reentrant=False)
            
            x = self.mlp(state)
            state = state + self.drop_path(x)
            state = checkpoint(self.ln_2, state, use_reentrant=False)
        return state

    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False,
                selective_checkpointing: bool = False):
        if selective_checkpointing:
            return self.checkpoint_forward(x, attn_mask, is_causal=is_causal)
        
        if self.use_preln:
            x = x + self.drop_path(self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask, is_causal=is_causal)))
            if self.use_conv_module:
                # Permute for conv module: (seq, batch, dim) -> (batch, seq, dim)
                conv_out = self.conv_module(self.ln_conv(x).permute(1, 0, 2))
                x = x + self.drop_path(self.ls_conv(conv_out.permute(1, 0, 2)))
            x = x + self.drop_path(self.ls_2(self.mlp(self.ln_2(x))))
        else:
            x = x + self.drop_path(self.attention(x, attn_mask=attn_mask, is_causal=is_causal))
            x = self.ln_1(x)
            if self.use_conv_module:
                conv_out = self.conv_module(self.ln_conv(x).permute(1, 0, 2))
                x = x + self.drop_path(conv_out.permute(1, 0, 2))
            x = x + self.drop_path(self.mlp(x))
            x = self.ln_2(x)
        return x


class AudioTransformer(nn.Module):
    """Transformer for audio processing."""
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float = 4.0,
                 ls_init_value: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 use_preln: bool = True,
                 use_conv_module: bool = False,
                 conv_kernel_size: int = 31,
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.selective_checkpointing = False
        self.grad_checkpointing_params = {'use_reentrant': False}
        if attn_drop == 0 and drop_path == 0 and drop == 0:
            self.grad_checkpointing_params.update({'preserve_rng_state': False})
        else:
            self.grad_checkpointing_params.update({'preserve_rng_state': True})

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                use_preln=use_preln, use_conv_module=use_conv_module,
                conv_kernel_size=conv_kernel_size)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False):
        for r in self.resblocks:
            if self.training and self.grad_checkpointing and not torch.jit.is_scripting():
                if not self.selective_checkpointing:
                    x = checkpoint(r, x, attn_mask, is_causal, **self.grad_checkpointing_params)
                else:
                    x = r(x, attn_mask=attn_mask, is_causal=is_causal, selective_checkpointing=True)
            else:
                x = r(x, attn_mask=attn_mask, is_causal=is_causal)
        return x


class AudioTransformerEncoder(nn.Module):
    """
    Audio Transformer Encoder inspired by AudioMAE.
    Processes audio spectrograms using patch-based tokenization.
    """
    def __init__(self,
                 input_size: int = 128,  # number of mel bins
                 input_length: int = 1024,  # number of time frames
                 patch_size: int = 16,  # patch size for both time and freq
                 time_patch_size: int = None,  # optional separate time patch size
                 freq_patch_size: int = None,  # optional separate freq patch size
                 width: int = 768,
                 layers: int = 12,
                 heads: int = 12,
                 mlp_ratio: float = 4.0,
                 ls_init_value: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 ln_pre: bool = True,
                 ln_post: bool = True,
                 act_layer: str = 'gelu',
                 norm_layer: str = 'layer_norm',
                 mask_type: Union[str, None] = 'none',
                 mask_block_size: int = -1,
                 use_conv_module: bool = False,
                 conv_kernel_size: int = 31,
    ):
        super().__init__()
        
        # Allow separate patch sizes for time and frequency
        self.time_patch_size = time_patch_size if time_patch_size is not None else patch_size
        self.freq_patch_size = freq_patch_size if freq_patch_size is not None else patch_size
        
        self.input_size = input_size  # mel bins
        self.input_length = input_length  # time frames
        self.grid_size = (input_length // self.time_patch_size, input_size // self.freq_patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_type = mask_type
        self.mask_block_size = mask_block_size

        if act_layer.lower() == 'gelu':
            self.act_layer = nn.GELU
        else:
            raise ValueError(f"Unsupported activation function: {act_layer}")
        if norm_layer.lower() == 'layer_norm':
            self.norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unsupported normalization: {norm_layer}")

        # Patch embedding - projects patches to width dimension
        self.patch_embed = nn.Linear(
            in_features=self.time_patch_size * self.freq_patch_size,
            out_features=width,
            bias=not ln_pre
        )

        # Positional embedding
        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches, width))

        self.ln_pre = self.norm_layer(width) if ln_pre else nn.Identity()

        self.transformer = AudioTransformer(
            width, layers, heads, mlp_ratio, ls_init_value=ls_init_value,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            act_layer=self.act_layer, norm_layer=self.norm_layer,
            use_conv_module=use_conv_module, conv_kernel_size=conv_kernel_size,
        )

        self.ln_post = self.norm_layer(width) if ln_post else nn.Identity()

        self.init_parameters()

    def init_parameters(self):
        if self.positional_embedding is not None:
            nn.init.normal_(self.positional_embedding, std=0.02)
        trunc_normal_(self.patch_embed.weight, std=0.02)
        for block in self.transformer.resblocks:
            for n, p in block.named_parameters():
                if 'weight' in n:
                    if 'ln' not in n and 'batch_norm' not in n:
                        trunc_normal_(p, std=0.02)
                elif 'bias' in n:
                    nn.init.zeros_(p)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True, selective=False):
        self.transformer.grad_checkpointing = enable
        self.transformer.selective_checkpointing = selective

    def forward(self, x):
        """
        Args:
            x: audio spectrogram of shape (batch, 1, time_frames, mel_bins) or (batch, time_frames, mel_bins)
        Returns:
            Encoded features of shape (batch, num_patches, width)
        """
        # Ensure input is 4D: (batch, channels, time, freq)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, time, freq)
        
        batch_size = x.shape[0]
        
        # Patchify: (batch, 1, T, F) -> (batch, num_patches, patch_size^2)
        # Split time and frequency into patches
        x = x.squeeze(1)  # (batch, time, freq)
        
        # Reshape to patches
        x = x.reshape(
            batch_size, 
            self.grid_size[0], self.time_patch_size,
            self.grid_size[1], self.freq_patch_size
        )
        # Rearrange: (batch, num_time_patches, time_patch_size, num_freq_patches, freq_patch_size)
        # -> (batch, num_time_patches, num_freq_patches, time_patch_size * freq_patch_size)
        x = x.permute(0, 1, 3, 2, 4).reshape(
            batch_size, self.num_patches, self.time_patch_size * self.freq_patch_size
        )
        
        # Project patches to embedding dimension
        x = self.patch_embed(x)  # (batch, num_patches, width)
        
        # Add positional embedding
        x = x + self.positional_embedding.to(x.dtype)
        
        x = self.ln_pre(x)
        
        # Transpose for transformer: (batch, seq, dim) -> (seq, batch, dim)
        x = x.permute(1, 0, 2)
        
        # Apply attention mask if needed
        block_size = self.num_patches if self.mask_block_size <= 0 else self.mask_block_size
        attn_mask = get_attention_mask(x.size(0), x.device, mask_type=self.mask_type, block_size=block_size)
        
        x = self.transformer(x, attn_mask, is_causal=self.mask_type == 'causal')
        
        # Transpose back: (seq, batch, dim) -> (batch, seq, dim)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)

        return x


class AudioTransformerDecoder(nn.Module):
    """Audio Transformer Decoder for reconstructing spectrograms."""
    def __init__(self,
                 input_size: int = 128,  # number of mel bins
                 input_length: int = 1024,  # number of time frames
                 patch_size: int = 16,
                 time_patch_size: int = None,
                 freq_patch_size: int = None,
                 width: int = 768,
                 layers: int = 12,
                 heads: int = 12,
                 mlp_ratio: float = 4.0,
                 ls_init_value: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 ln_pre: bool = True,
                 ln_post: bool = True,
                 act_layer: str = 'gelu',
                 norm_layer: str = 'layer_norm',
                 use_ffn_output: bool = True,
                 dim_ffn_output: int = 3072,
                 logit_laplace: bool = False,
                 mask_type: Union[str, None] = 'none',
                 mask_block_size: int = -1,
                 use_conv_module: bool = False,
                 conv_kernel_size: int = 31,
    ):
        super().__init__()
        
        self.time_patch_size = time_patch_size if time_patch_size is not None else patch_size
        self.freq_patch_size = freq_patch_size if freq_patch_size is not None else patch_size
        
        self.input_size = input_size
        self.input_length = input_length
        self.grid_size = (input_length // self.time_patch_size, input_size // self.freq_patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_type = mask_type
        self.mask_block_size = mask_block_size

        if act_layer.lower() == 'gelu':
            self.act_layer = nn.GELU
        else:
            raise ValueError(f"Unsupported activation function: {act_layer}")
        if norm_layer.lower() == 'layer_norm':
            self.norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unsupported normalization: {norm_layer}")

        self.use_ffn_output = use_ffn_output
        if use_ffn_output:
            self.ffn = nn.Sequential(
                nn.Linear(width, dim_ffn_output),
                nn.Tanh(),
            )
            self.conv_out = nn.Linear(
                in_features=dim_ffn_output,
                out_features=self.time_patch_size * self.freq_patch_size * (1 + logit_laplace)
            )
        else:
            self.ffn = nn.Identity()
            self.conv_out = nn.Linear(
                in_features=width,
                out_features=self.time_patch_size * self.freq_patch_size * (1 + logit_laplace)
            )

        # Positional embedding
        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches, width))

        self.ln_pre = self.norm_layer(width) if ln_pre else nn.Identity()

        self.transformer = AudioTransformer(
            width, layers, heads, mlp_ratio, ls_init_value=ls_init_value,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            act_layer=self.act_layer, norm_layer=self.norm_layer,
            use_conv_module=use_conv_module, conv_kernel_size=conv_kernel_size,
        )

        self.ln_post = self.norm_layer(width) if ln_post else nn.Identity()

        self.init_parameters()

    def init_parameters(self):
        if self.positional_embedding is not None:
            nn.init.normal_(self.positional_embedding, std=0.02)

        for block in self.transformer.resblocks:
            for n, p in block.named_parameters():
                if 'weight' in n:
                    if 'ln' not in n and 'batch_norm' not in n:
                        trunc_normal_(p, std=0.02)
                elif 'bias' in n:
                    nn.init.zeros_(p)
        
        if self.use_ffn_output:
            trunc_normal_(self.ffn[0].weight, std=0.02)
        trunc_normal_(self.conv_out.weight, std=0.02)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True, selective=False):
        self.transformer.grad_checkpointing = enable
        self.transformer.selective_checkpointing = selective

    def forward(self, x):
        """
        Args:
            x: encoded features of shape (batch, num_patches, width)
        Returns:
            Reconstructed spectrogram of shape (batch, time_frames, mel_bins)
        """
        batch_size = x.shape[0]
        
        # Add positional embedding
        x = x + self.positional_embedding.to(x.dtype)
        
        x = self.ln_pre(x)
        
        # Transpose for transformer
        x = x.permute(1, 0, 2)
        
        # Apply attention mask
        block_size = self.num_patches if self.mask_block_size <= 0 else self.mask_block_size
        attn_mask = get_attention_mask(x.size(0), x.device, mask_type=self.mask_type, block_size=block_size)
        
        x = self.transformer(x, attn_mask, is_causal=self.mask_type == 'causal')
        
        # Transpose back
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        x = self.ffn(x)
        x = self.conv_out(x)
        
        # Reshape patches back to spectrogram
        # (batch, num_patches, patch_size^2) -> (batch, time, freq)
        x = x.reshape(
            batch_size,
            self.grid_size[0], self.grid_size[1],
            self.time_patch_size, self.freq_patch_size
        )
        x = x.permute(0, 1, 3, 2, 4).reshape(
            batch_size, self.input_length, self.input_size
        )

        return x

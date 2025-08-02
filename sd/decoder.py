"""
VAE Decoder for Stable Diffusion.

This module implements the decoder component of the Variational Autoencoder (VAE)
used in Stable Diffusion. The decoder transforms latent representations from the
diffusion process back into RGB images.

The decoder architecture consists of:
1. Residual blocks for feature processing
2. Attention blocks for spatial relationships
3. Upsampling layers to increase spatial resolution
4. Final convolution to produce RGB output

The decoder takes 4-channel latents at 1/8 resolution and outputs 3-channel RGB
images at full resolution (8x upsampling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    """
    Self-attention block for VAE decoder.
    
    This block applies self-attention across spatial locations to help the model
    understand relationships between different parts of the image. Each pixel
    is treated as a token in the attention mechanism.
    
    Args:
        channels (int): Number of input/output channels
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Group normalization for stable training
        # 32 groups works well for typical channel counts (512, 256, 128)
        self.groupnorm = nn.GroupNorm(32, channels)
        
        # Single-head self-attention (sufficient for spatial relationships)
        self.attention = SelfAttention(n_heads=1, d_embed=channels)
        
        self.channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention block.
        
        Args:
            x (torch.Tensor): Input feature map of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output feature map of same shape
        """
        # Store original input for residual connection
        residual = x
        batch_size, channels, height, width = x.shape

        # === Step 1: Normalize input ===
        x = self.groupnorm(x)
        
        # === Step 2: Reshape for attention ===
        # Convert spatial dimensions to sequence length for attention
        # (batch_size, channels, height, width) -> (batch_size, channels, height*width)
        x = x.view(batch_size, channels, height * width)
        
        # Transpose to get sequence-first format for attention
        # (batch_size, channels, height*width) -> (batch_size, height*width, channels)
        x = x.transpose(-1, -2)
        
        # === Step 3: Apply self-attention ===
        # Each pixel attends to all other pixels (no causal masking for images)
        x = self.attention(x, causal_mask=False)
        
        # === Step 4: Reshape back to spatial format ===
        # (batch_size, height*width, channels) -> (batch_size, channels, height*width)
        x = x.transpose(-1, -2)
        
        # (batch_size, channels, height*width) -> (batch_size, channels, height, width)
        x = x.view(batch_size, channels, height, width)
        
        # === Step 5: Add residual connection ===
        x = x + residual

        return x 

class VAE_ResidualBlock(nn.Module):
    """
    Residual block for VAE decoder.
    
    This block implements a standard residual connection with:
    - Group normalization for stable training
    - SiLU (Swish) activation for smooth gradients
    - Two 3x3 convolutions
    - Optional channel adjustment via 1x1 convolution
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # First normalization and convolution
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Second normalization and convolution
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection handling
        if in_channels == out_channels:
            # No channel adjustment needed
            self.residual_layer = nn.Identity()
        else:
            # Adjust channels with 1x1 convolution
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Store input for residual connection
        residual = x

        # === First convolution block ===
        x = self.groupnorm_1(x)
        x = F.silu(x)  # SiLU activation: x * sigmoid(x)
        x = self.conv_1(x)
        
        # === Second convolution block ===
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        
        # === Add residual connection ===
        # Apply channel adjustment if needed
        return x + self.residual_layer(residual)

class VAE_Decoder(nn.Sequential):
    """
    VAE Decoder for Stable Diffusion.
    
    This decoder transforms 4-channel latent representations (at 1/8 resolution)
    back into 3-channel RGB images (at full resolution). The architecture follows
    a progressive upsampling approach:
    
    1. Initial convolutions at 1/8 resolution (512 channels)
    2. Residual blocks and attention for feature processing
    3. Upsampling stages: 1/8 -> 1/4 -> 1/2 -> 1x resolution
    4. Channel reduction: 512 -> 256 -> 128 -> 3 channels
    5. Final RGB output
    
    The scaling factor 0.18215 is applied to normalize the latent space as per
    the original Stable Diffusion implementation.
    """
    
    def __init__(self):
        super().__init__(
            # === Initial Processing (1/8 resolution) ===
            # Identity convolution for potential learned scaling
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # Expand to working channel dimension
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # === Deep Feature Processing ===
            # Multiple residual blocks for complex feature extraction
            VAE_ResidualBlock(512, 512), 
            
            # Self-attention to capture spatial relationships
            VAE_AttentionBlock(512), 
            
            # Additional residual processing
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # === First Upsampling Stage: 1/8 -> 1/4 resolution ===
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # Feature refinement at 1/4 resolution
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # === Second Upsampling Stage: 1/4 -> 1/2 resolution ===
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # Channel reduction and feature refinement
            VAE_ResidualBlock(512, 256), 
            VAE_ResidualBlock(256, 256), 
            VAE_ResidualBlock(256, 256), 
            
            # === Third Upsampling Stage: 1/2 -> 1x resolution ===
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            
            # Further channel reduction
            VAE_ResidualBlock(256, 128), 
            VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128), 
            
            # === Final Output Processing ===
            # Final normalization and activation
            nn.GroupNorm(32, 128), 
            nn.SiLU(), 
            
            # Convert to RGB output
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of VAE decoder.
        
        Args:
            x (torch.Tensor): Latent tensor of shape (batch_size, 4, height/8, width/8)
            
        Returns:
            torch.Tensor: RGB image tensor of shape (batch_size, 3, height, width)
        """
        # Remove the scaling applied by the encoder
        # This value (0.18215) is specific to Stable Diffusion's latent space normalization
        x = x / 0.18215

        # Apply all decoder layers sequentially
        for module in self:
            x = module(x)

        return x
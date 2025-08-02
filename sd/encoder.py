"""
VAE Encoder for Stable Diffusion.

This module implements the encoder component of the Variational Autoencoder (VAE)
used in Stable Diffusion. The encoder compresses RGB images into a compact latent
representation that can be efficiently processed by the diffusion model.

The encoder architecture follows a progressive downsampling approach:
1. Initial feature extraction from RGB input
2. Progressive downsampling: 1x -> 1/2 -> 1/4 -> 1/8 resolution
3. Channel expansion: 3 -> 128 -> 256 -> 512 channels
4. Deep feature processing with residual blocks and attention
5. Variational encoding to latent space (mean and log-variance)
6. Reparameterization trick for sampling

The final output is 4-channel latents at 1/8 resolution, providing an 8x8 = 64x
compression ratio in spatial dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    """
    VAE Encoder for compressing images to latent space.
    
    This encoder transforms RGB images into compact latent representations
    through progressive downsampling and feature extraction. The architecture
    uses residual blocks for deep feature processing and self-attention for
    capturing spatial relationships.
    
    Architecture Overview:
    - Input: RGB images (3 channels, full resolution)
    - Progressive downsampling: 3 stages with 2x reduction each
    - Feature extraction: Multiple residual blocks per stage
    - Attention: Self-attention at the deepest level
    - Output: Latent parameters (8 channels -> mean + log_variance)
    - Final: Sampled latents (4 channels at 1/8 resolution)
    
    The 0.18215 scaling factor is applied to normalize the latent space
    as per the original Stable Diffusion configuration.
    """
    
    def __init__(self):
        super().__init__(
            # === Stage 1: Initial Feature Extraction (Full Resolution) ===
            # Convert RGB to feature representation
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # Deep feature processing at full resolution
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            # === Stage 2: First Downsampling (1x -> 1/2 Resolution) ===
            # Downsample with stride=2, padding handled in forward pass
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # Feature processing and channel expansion
            VAE_ResidualBlock(128, 256), 
            VAE_ResidualBlock(256, 256), 
            
            # === Stage 3: Second Downsampling (1/2 -> 1/4 Resolution) ===
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            # Feature processing and channel expansion
            VAE_ResidualBlock(256, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # === Stage 4: Third Downsampling (1/4 -> 1/8 Resolution) ===
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            # === Stage 5: Deep Feature Processing at 1/8 Resolution ===
            # Multiple residual blocks for complex feature extraction
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # Self-attention for spatial relationship modeling
            VAE_AttentionBlock(512), 
            
            # Final residual processing
            VAE_ResidualBlock(512, 512), 
            
            # === Stage 6: Variational Encoding ===
            # Final normalization and activation
            nn.GroupNorm(32, 512), 
            nn.SiLU(), 

            # Convert to latent parameter space (mean and log-variance)
            # 8 channels = 4 for mean + 4 for log-variance
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            # Optional refinement convolution
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of VAE encoder.
        
        This method processes RGB images through the encoding pipeline and applies
        the reparameterization trick to sample from the learned latent distribution.
        
        Args:
            x (torch.Tensor): Input RGB images of shape (batch_size, 3, height, width)
            noise (torch.Tensor): Random noise for reparameterization of shape (batch_size, 4, height/8, width/8)
            
        Returns:
            torch.Tensor: Encoded latent representation of shape (batch_size, 4, height/8, width/8)
        """
        # === Progressive encoding through network layers ===
        for module in self:
            # Handle asymmetric padding for downsampling layers
            if getattr(module, 'stride', None) == (2, 2):
                # Apply asymmetric padding for proper downsampling
                # Pad: (left, right, top, bottom) = (0, 1, 0, 1)
                # This ensures proper spatial alignment during downsampling
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        
        # === Variational latent space processing ===
        # Split the 8-channel output into mean and log-variance
        # Shape: (batch_size, 8, height/8, width/8) -> 2x (batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # === Variance stabilization ===
        # Clamp log-variance to prevent numerical instability
        # Range [-30, 20] corresponds to variance range [~1e-14, ~1e8]
        log_variance = torch.clamp(log_variance, -30, 20)
        
        # Convert log-variance to variance and standard deviation
        variance = log_variance.exp()
        stdev = variance.sqrt()
        
        # === Reparameterization trick ===
        # Sample from N(mean, stdev) using N(0,1) noise
        # This allows gradients to flow through the sampling process
        # Formula: z = μ + σ * ε, where ε ~ N(0,1)
        x = mean + stdev * noise
        
        # === Latent space normalization ===
        # Apply scaling factor to normalize latent space
        # Value 0.18215 is from original Stable Diffusion configuration
        # Reference: https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml#L17
        x *= 0.18215
        
        return x


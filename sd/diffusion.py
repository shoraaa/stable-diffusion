"""
U-Net Diffusion Model for Stable Diffusion.

This module implements the U-Net architecture used in Stable Diffusion for
the denoising process. The U-Net is responsible for predicting noise to be
removed from latent representations at each diffusion timestep.

Key Components:
1. Time Embedding: Encodes timestep information
2. U-Net Encoder: Downsamples latents while extracting features
3. Bottleneck: Processes features at lowest resolution
4. U-Net Decoder: Upsamples while incorporating skip connections
5. Cross-Attention: Conditions generation on text embeddings

The architecture follows a symmetric encoder-decoder structure with
skip connections and incorporates both self-attention and cross-attention
mechanisms for spatial and conditional relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    """
    Time embedding layer for diffusion timesteps.
    
    Converts scalar timestep values into rich embeddings that inform
    the U-Net about the current noise level in the diffusion process.
    
    Args:
        n_embd (int): Input embedding dimension (typically 320)
    """
    
    def __init__(self, n_embd: int):
        super().__init__()
        
        # Expand timestep embedding to higher dimension
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        
        # Further process the expanded embedding
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)
        
        self.n_embd = n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of time embedding.
        
        Args:
            x (torch.Tensor): Timestep embedding of shape (1, n_embd)
            
        Returns:
            torch.Tensor: Processed time embedding of shape (1, 4*n_embd)
        """
        # Expand embedding dimension: (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        
        # Apply SiLU activation for smooth gradients
        x = F.silu(x)
        
        # Further process the embedding: (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x

class UNET_ResidualBlock(nn.Module):
    """
    Residual block for U-Net with time conditioning.
    
    This block combines spatial feature processing with temporal information
    from the diffusion timestep. It follows the ResNet pattern with two
    convolutions and includes time injection between them.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels  
        n_time (int): Dimension of time embedding. Defaults to 1280.
    """
    
    def __init__(self, in_channels: int, out_channels: int, n_time: int = 1280):
        super().__init__()
        
        # First convolution path
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time conditioning
        self.linear_time = nn.Linear(n_time, out_channels)

        # Second convolution path  
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection handling
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            # Channel adjustment for residual connection
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of U-Net residual block.
        
        Args:
            feature (torch.Tensor): Spatial features of shape (batch_size, in_channels, height, width)
            time (torch.Tensor): Time embedding of shape (1, n_time)
            
        Returns:
            torch.Tensor: Output features of shape (batch_size, out_channels, height, width)
        """
        # Store input for residual connection
        residual = feature
        
        # === First convolution block ===
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        
        # === Time conditioning ===
        # Process time embedding
        time = F.silu(time)
        time = self.linear_time(time)
        
        # Inject time information by broadcasting to spatial dimensions
        # Add singleton dimensions for height and width: (1, out_channels) -> (1, out_channels, 1, 1)
        # This allows broadcasting: (batch_size, out_channels, height, width) + (1, out_channels, 1, 1)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # === Second convolution block ===
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        
        # === Residual connection ===
        return merged + self.residual_layer(residual)

class UNET_AttentionBlock(nn.Module):
    """
    Transformer-style attention block for U-Net.
    
    This block implements a complete transformer layer with:
    1. Self-attention for spatial relationships 
    2. Cross-attention for text conditioning
    3. GeGLU feedforward network
    
    Each sub-layer includes layer normalization and residual connections.
    
    Args:
        n_head (int): Number of attention heads
        n_embd (int): Embedding dimension per head
        d_context (int): Dimension of context (text) embeddings. Defaults to 768.
    """
    
    def __init__(self, n_head: int, n_embd: int, d_context: int = 768):
        super().__init__()
        channels = n_head * n_embd
        
        # Input/output convolutions for spatial processing
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # Self-attention sub-layer
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        
        # Cross-attention sub-layer  
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        
        # GeGLU feedforward sub-layer
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)  # 2x for gate
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.n_head = n_head
        self.n_embd = n_embd
        self.channels = channels
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of U-Net attention block.
        
        Args:
            x (torch.Tensor): Spatial features of shape (batch_size, channels, height, width)
            context (torch.Tensor): Text context of shape (batch_size, seq_len, d_context)
            
        Returns:
            torch.Tensor: Output features of same spatial shape as input
        """
        # Store input for long residual connection
        residual_long = x
        batch_size, channels, height, width = x.shape

        # === Input processing ===
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        # === Reshape for attention ===
        # Convert to sequence format: (batch_size, channels, height*width)
        x = x.view(batch_size, channels, height * width)
        # Transpose for attention: (batch_size, height*width, channels)
        x = x.transpose(-1, -2)
        
        # === Self-attention sub-layer ===
        residual_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x, causal_mask=False)  # No causal masking for images
        x = x + residual_short

        # === Cross-attention sub-layer ===
        residual_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)  # Condition on text embeddings
        x = x + residual_short

        # === GeGLU feedforward sub-layer ===
        residual_short = x
        x = self.layernorm_3(x)
        
        # GeGLU: Split into value and gate, apply GELU to gate
        # Reference: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/attention.py#L37
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)  # Gated Linear Unit with GELU
        x = self.linear_geglu_2(x)
        x = x + residual_short
        
        # === Reshape back to spatial format ===
        x = x.transpose(-1, -2)  # (batch_size, channels, height*width)
        x = x.view(batch_size, channels, height, width)

        # === Output processing with long residual connection ===
        return self.conv_output(x) + residual_long

class Upsample(nn.Module):
    """
    Upsampling layer for U-Net decoder.
    
    Doubles the spatial resolution using nearest neighbor interpolation
    followed by a convolution to refine the upsampled features.
    
    Args:
        channels (int): Number of input/output channels
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of upsampling layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Upsampled tensor of shape (batch_size, channels, height*2, width*2)
        """
        # Upsample by factor of 2 using nearest neighbor interpolation
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        # Refine upsampled features with convolution
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    """
    Sequential container that handles different layer types in U-Net.
    
    This container automatically routes inputs based on layer type:
    - UNET_AttentionBlock: receives (x, context)
    - UNET_ResidualBlock: receives (x, time)  
    - Other layers: receive only x
    """
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic input routing.
        
        Args:
            x (torch.Tensor): Spatial features
            context (torch.Tensor): Text context embeddings
            time (torch.Tensor): Time embeddings
            
        Returns:
            torch.Tensor: Processed features
        """
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                # Standard layers (Conv2d, Upsample, etc.)
                x = layer(x)
        return x

class UNET(nn.Module):
    """
    U-Net architecture for diffusion denoising.
    
    This implements the core U-Net used in Stable Diffusion for predicting
    noise to be removed from latent representations. The architecture features:
    
    1. Encoder: Progressive downsampling with attention
    2. Bottleneck: Processing at lowest resolution
    3. Decoder: Progressive upsampling with skip connections
    
    Resolution progression:
    - Input: 1/8 resolution (64x64 for 512x512 image)
    - Encoder: 1/8 -> 1/16 -> 1/32 -> 1/64
    - Decoder: 1/64 -> 1/32 -> 1/16 -> 1/8 -> 1/8 (output)
    
    Channel progression:
    - 4 -> 320 -> 640 -> 1280 -> 1280 (encoder)
    - 1280 -> 640 -> 320 -> 4 (decoder)
    """
    
    def __init__(self):
        super().__init__()
        
        # === ENCODER: Progressive downsampling ===
        self.encoders = nn.ModuleList([
            # Stage 1: Initial convolution (1/8 resolution)
            # 4 -> 320 channels
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            # Stage 2: Process at 1/8 resolution  
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # Stage 3: Downsample to 1/16 resolution
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # Stage 4: Process at 1/16 resolution, expand to 640 channels
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # Stage 5: Downsample to 1/32 resolution  
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # Stage 6: Process at 1/32 resolution, expand to 1280 channels
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # Stage 7: Downsample to 1/64 resolution (bottleneck level)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # Stage 8: Deep processing at 1/64 resolution
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        # === BOTTLENECK: Processing at lowest resolution (1/64) ===
        self.bottleneck = SwitchSequential(
            # Deep feature processing with attention at bottleneck
            UNET_ResidualBlock(1280, 1280), 
            UNET_AttentionBlock(8, 160), 
            UNET_ResidualBlock(1280, 1280), 
        )
        
        # === DECODER: Progressive upsampling with skip connections ===
        self.decoders = nn.ModuleList([
            # Stage 1: Process concatenated features at 1/64 resolution
            # 2560 channels = 1280 (bottleneck) + 1280 (skip connection)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # Stage 2: Upsample to 1/32 resolution
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # Stage 3: Process at 1/32 resolution with attention
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # Stage 4: Upsample to 1/16 resolution
            # 1920 channels = 1280 (current) + 640 (skip connection)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # Stage 5: Process at 1/16 resolution, reduce to 640 channels
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # Stage 6: Upsample to 1/8 resolution
            # 960 channels = 640 (current) + 320 (skip connection)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # Stage 7: Process at 1/8 resolution, reduce to 320 channels
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of U-Net.
        
        Args:
            x (torch.Tensor): Input latents of shape (batch_size, 4, height/8, width/8)
            context (torch.Tensor): Text embeddings of shape (batch_size, seq_len, 768)
            time (torch.Tensor): Time embeddings of shape (1, 1280)
            
        Returns:
            torch.Tensor: Processed features of shape (batch_size, 320, height/8, width/8)
        """
        # === Encoder pass with skip connection storage ===
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x, context, time)
            skip_connections.append(x)

        # === Bottleneck processing ===
        x = self.bottleneck(x, context, time)

        # === Decoder pass with skip connections ===
        for decoder in self.decoders:
            # Concatenate with corresponding encoder features (skip connection)
            # This provides both low-level and high-level information to the decoder
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = decoder(x, context, time)
        
        return x


class UNET_OutputLayer(nn.Module):
    """
    Final output layer for U-Net.
    
    Converts U-Net features back to latent space format for the diffusion process.
    This layer produces the predicted noise that should be removed from the input.
    
    Args:
        in_channels (int): Number of input channels (typically 320)
        out_channels (int): Number of output channels (typically 4 for latent space)
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of output layer.
        
        Args:
            x (torch.Tensor): U-Net features of shape (batch_size, in_channels, height/8, width/8)
            
        Returns:
            torch.Tensor: Predicted noise of shape (batch_size, out_channels, height/8, width/8)
        """
        # Normalize features
        x = self.groupnorm(x)
        
        # Apply activation
        x = F.silu(x)
        
        # Convert to output space (predicted noise)
        x = self.conv(x)
        
        return x


class Diffusion(nn.Module):
    """
    Complete diffusion model for Stable Diffusion.
    
    This module combines all components needed for the diffusion denoising process:
    1. Time embedding: Processes timestep information
    2. U-Net: Core denoising network
    3. Output layer: Converts features to predicted noise
    
    The model takes noisy latents, text context, and timestep information
    to predict the noise that should be removed.
    """
    
    def __init__(self):
        super().__init__()
        
        # Time embedding processor
        self.time_embedding = TimeEmbedding(320)
        
        # Core U-Net denoising network
        self.unet = UNET()
        
        # Final output layer
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(
        self, 
        latent: torch.Tensor, 
        context: torch.Tensor, 
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of diffusion model.
        
        Args:
            latent (torch.Tensor): Noisy latent representation of shape (batch_size, 4, height/8, width/8)
            context (torch.Tensor): Text conditioning embeddings of shape (batch_size, seq_len, 768)
            time (torch.Tensor): Timestep embedding of shape (1, 320)
            
        Returns:
            torch.Tensor: Predicted noise of shape (batch_size, 4, height/8, width/8)
        """
        # === Process timestep information ===
        # Convert timestep to rich embedding: (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # === Apply U-Net denoising ===
        # Process noisy latents with text conditioning and time information
        output = self.unet(latent, context, time)
        
        # === Generate final prediction ===
        # Convert U-Net features to predicted noise
        output = self.final(output)
        
        return output
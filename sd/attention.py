"""
Attention mechanisms for Stable Diffusion model.

This module implements both self-attention and cross-attention mechanisms
used in the Stable Diffusion architecture. These attention layers are
fundamental components for processing and relating information across
different parts of the input sequence.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    This implementation follows the standard transformer self-attention pattern
    where each token attends to all other tokens in the same sequence.
    
    Args:
        n_heads (int): Number of attention heads
        d_embed (int): Embedding dimension
        in_proj_bias (bool): Whether to use bias in input projection. Defaults to True.
        out_proj_bias (bool): Whether to use bias in output projection. Defaults to True.
    """
    
    def __init__(
        self, 
        n_heads: int, 
        d_embed: int, 
        in_proj_bias: bool = True, 
        out_proj_bias: bool = True
    ):
        super().__init__()
        
        # Combined linear projection for Q, K, V matrices (more efficient than separate projections)
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        
        # Output projection matrix (Wo in the attention paper)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"
    
    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        """
        Forward pass of self-attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_embed)
            causal_mask (bool): Whether to apply causal masking (for autoregressive models)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_embed)
        """
        # Store input shape for later reshaping
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        
        # Shape for multi-head attention: (batch_size, seq_len, n_heads, d_head)
        multi_head_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        
        # === Step 1: Generate Q, K, V matrices ===
        # Project input to Q, K, V and split into three tensors
        # Shape: (batch_size, seq_len, d_embed) -> (batch_size, seq_len, 3*d_embed) -> 3x(batch_size, seq_len, d_embed)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # === Step 2: Reshape for multi-head attention ===
        # Reshape and transpose to get shape: (batch_size, n_heads, seq_len, d_head)
        q = q.view(multi_head_shape).transpose(1, 2)
        k = k.view(multi_head_shape).transpose(1, 2)
        v = v.view(multi_head_shape).transpose(1, 2)
        
        # === Step 3: Compute attention scores ===
        # Calculate attention weights: Q @ K^T
        # Shape: (batch_size, n_heads, seq_len, d_head) @ (batch_size, n_heads, d_head, seq_len)
        #     -> (batch_size, n_heads, seq_len, seq_len)
        attention_scores = q @ k.transpose(-1, -2)
        
        # === Step 4: Apply causal masking if requested ===
        if causal_mask:
            # Create upper triangular mask to prevent attending to future tokens
            mask = torch.ones_like(attention_scores, dtype=torch.bool).triu(1)
            attention_scores.masked_fill_(mask, -torch.inf)
        
        # === Step 5: Scale attention scores ===
        # Scale by 1/sqrt(d_head) as per "Attention Is All You Need"
        attention_scores /= math.sqrt(self.d_head)
        
        # === Step 6: Apply softmax to get attention weights ===
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # === Step 7: Apply attention to values ===
        # Weighted sum of values using attention weights
        # Shape: (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_head)
        #     -> (batch_size, n_heads, seq_len, d_head)
        output = attention_weights @ v
        
        # === Step 8: Reshape and project output ===
        # Transpose back: (batch_size, n_heads, seq_len, d_head) -> (batch_size, seq_len, n_heads, d_head)
        output = output.transpose(1, 2)
        
        # Reshape to original embedding dimension: (batch_size, seq_len, d_embed)
        output = output.reshape(input_shape)
        
        # Final linear projection
        output = self.out_proj(output)
        
        return output


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism.
    
    Cross-attention allows one sequence (queries) to attend to another sequence
    (keys and values). This is commonly used in encoder-decoder architectures
    and in Stable Diffusion for conditioning the image generation on text embeddings.
    
    Args:
        n_heads (int): Number of attention heads
        d_embed (int): Embedding dimension for queries
        d_cross (int): Embedding dimension for keys and values (context)
        in_proj_bias (bool): Whether to use bias in input projections. Defaults to True.
        out_proj_bias (bool): Whether to use bias in output projection. Defaults to True.
    """
    
    def __init__(
        self, 
        n_heads: int, 
        d_embed: int, 
        d_cross: int, 
        in_proj_bias: bool = True, 
        out_proj_bias: bool = True
    ):
        super().__init__()
        
        # Separate projections for Q, K, V (different input dimensions for K, V)
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)  # Query projection
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)  # Key projection
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)  # Value projection
        
        # Output projection matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of cross-attention.
        
        Args:
            x (torch.Tensor): Query tensor (latent features) of shape (batch_size, seq_len_q, d_embed)
            y (torch.Tensor): Key-Value tensor (context/conditioning) of shape (batch_size, seq_len_kv, d_cross)
                             In Stable Diffusion, this is typically text embeddings with shape (batch_size, 77, 768)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len_q, d_embed)
        """
        # Store input shape for later reshaping
        input_shape = x.shape
        batch_size, seq_len_q, d_embed = input_shape
        
        # Shape for multi-head attention: (batch_size, seq_len, n_heads, d_head)
        # Note: Using -1 for sequence length to handle different seq_len for K, V
        multi_head_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # === Step 1: Generate Q, K, V matrices ===
        # Project inputs to Q, K, V spaces
        q = self.q_proj(x)  # Shape: (batch_size, seq_len_q, d_embed)
        k = self.k_proj(y)  # Shape: (batch_size, seq_len_kv, d_embed)
        v = self.v_proj(y)  # Shape: (batch_size, seq_len_kv, d_embed)
        
        # === Step 2: Reshape for multi-head attention ===
        # Reshape and transpose to get shape: (batch_size, n_heads, seq_len, d_head)
        q = q.view(multi_head_shape).transpose(1, 2)  # (batch_size, n_heads, seq_len_q, d_head)
        k = k.view(multi_head_shape).transpose(1, 2)  # (batch_size, n_heads, seq_len_kv, d_head)
        v = v.view(multi_head_shape).transpose(1, 2)  # (batch_size, n_heads, seq_len_kv, d_head)
        
        # === Step 3: Compute cross-attention scores ===
        # Calculate attention weights: Q @ K^T
        # Shape: (batch_size, n_heads, seq_len_q, d_head) @ (batch_size, n_heads, d_head, seq_len_kv)
        #     -> (batch_size, n_heads, seq_len_q, seq_len_kv)
        attention_scores = q @ k.transpose(-1, -2)
        
        # === Step 4: Scale attention scores ===
        # Scale by 1/sqrt(d_head) for numerical stability
        attention_scores /= math.sqrt(self.d_head)
        
        # === Step 5: Apply softmax to get attention weights ===
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # === Step 6: Apply attention to values ===
        # Weighted sum of values using attention weights
        # Shape: (batch_size, n_heads, seq_len_q, seq_len_kv) @ (batch_size, n_heads, seq_len_kv, d_head)
        #     -> (batch_size, n_heads, seq_len_q, d_head)
        output = attention_weights @ v
        
        # === Step 7: Reshape and project output ===
        # Transpose back: (batch_size, n_heads, seq_len_q, d_head) -> (batch_size, seq_len_q, n_heads, d_head)
        output = output.transpose(1, 2).contiguous()
        
        # Reshape to original embedding dimension: (batch_size, seq_len_q, d_embed)
        output = output.view(input_shape)
        
        # Final linear projection
        output = self.out_proj(output)
        
        return output
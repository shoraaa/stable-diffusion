"""
CLIP (Contrastive Language-Image Pre-training) text encoder implementation.

This module implements the text encoder component of CLIP, which transforms
text tokens into embeddings that can be used for conditioning in Stable Diffusion.
The architecture follows the transformer-based text encoder from the original CLIP paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    """
    CLIP text embedding layer combining token and positional embeddings.
    
    This layer converts input token IDs to dense embeddings and adds positional
    information to encode the sequential nature of text.
    
    Args:
        n_vocab (int): Size of vocabulary (number of unique tokens)
        n_embd (int): Embedding dimension
        n_token (int): Maximum sequence length (number of positions)
    """
    
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        # Token embedding: maps vocabulary indices to dense vectors
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        
        # Positional embedding: learnable position encodings for each token position
        # Shape: (max_seq_len, embedding_dim)
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embd))
        
        self.n_token = n_token
        self.n_embd = n_embd
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass of CLIP embedding layer.
        
        Args:
            tokens (torch.LongTensor): Token IDs of shape (batch_size, seq_len)
            
        Returns:
            torch.FloatTensor: Embedded tokens of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len = tokens.shape
        
        # Ensure sequence length doesn't exceed maximum
        assert seq_len <= self.n_token, f"Sequence length {seq_len} exceeds maximum {self.n_token}"
        
        # Convert token IDs to embeddings
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, n_embd)
        token_embeddings = self.token_embedding(tokens)
        
        # Add positional embeddings (broadcast across batch dimension)
        # Shape: (batch_size, seq_len, n_embd) + (seq_len, n_embd) -> (batch_size, seq_len, n_embd)
        embeddings = token_embeddings + self.position_embedding[:seq_len]
        
        return embeddings

class CLIPLayer(nn.Module):
    """
    Single transformer layer for CLIP text encoder.
    
    Implements a standard transformer encoder layer with:
    - Layer normalization before self-attention (pre-norm architecture)
    - Multi-head self-attention with causal masking
    - Layer normalization before feedforward network
    - Feedforward network with QuickGELU activation
    - Residual connections around both sub-layers
    
    Args:
        n_head (int): Number of attention heads
        n_embd (int): Embedding dimension
    """
    
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        # Layer normalization before self-attention
        self.layernorm_1 = nn.LayerNorm(n_embd)
        
        # Multi-head self-attention mechanism
        self.attention = SelfAttention(n_head, n_embd)
        
        # Layer normalization before feedforward network
        self.layernorm_2 = nn.LayerNorm(n_embd)
        
        # Feedforward network: embedding_dim -> 4*embedding_dim -> embedding_dim
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)  # Expansion layer
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)   # Projection layer
        
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass of CLIP transformer layer.
        
        Args:
            x (torch.FloatTensor): Input embeddings of shape (batch_size, seq_len, n_embd)
            
        Returns:
            torch.FloatTensor: Output embeddings of shape (batch_size, seq_len, n_embd)
        """
        # === Self-Attention Block ===
        # Store residual connection
        residual = x
        
        # Pre-normalization before attention
        x = self.layernorm_1(x)
        
        # Apply self-attention with causal masking (for autoregressive text generation)
        x = self.attention(x, causal_mask=True)
        
        # Residual connection
        x = x + residual

        # === Feedforward Block ===
        # Store residual connection
        residual = x
        
        # Pre-normalization before feedforward
        x = self.layernorm_2(x)
        
        # First linear transformation (expansion)
        x = self.linear_1(x)
        
        # QuickGELU activation: x * sigmoid(1.702 * x)
        # This is a faster approximation of GELU used in the original CLIP
        x = x * torch.sigmoid(1.702 * x)
        
        # Second linear transformation (projection back to original dimension)
        x = self.linear_2(x)
        
        # Residual connection
        x = x + residual

        return x

class CLIP(nn.Module):
    """
    CLIP text encoder for Stable Diffusion.
    
    This is the complete text encoder that processes tokenized text and produces
    embeddings that can be used to condition the image generation process in
    Stable Diffusion. The architecture consists of:
    
    1. Token + positional embeddings
    2. Stack of transformer layers with self-attention
    3. Final layer normalization
    
    The default configuration matches the CLIP-ViT-L/14 text encoder:
    - Vocabulary size: 49,408
    - Embedding dimension: 768
    - Maximum sequence length: 77 tokens
    - 12 transformer layers
    - 12 attention heads per layer
    """
    
    def __init__(
        self,
        n_vocab: int = 49408,    # Size of vocabulary 
        n_embd: int = 768,       # Embedding dimension
        n_token: int = 77,       # Maximum sequence length
        n_layer: int = 12,       # Number of transformer layers
        n_head: int = 12         # Number of attention heads
    ):
        super().__init__()
        
        # Token and positional embedding layer
        self.embedding = CLIPEmbedding(n_vocab, n_embd, n_token)

        # Stack of transformer encoder layers
        self.layers = nn.ModuleList([
            CLIPLayer(n_head, n_embd) for _ in range(n_layer)
        ])

        # Final layer normalization
        self.layernorm = nn.LayerNorm(n_embd)
        
        # Store configuration
        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.n_token = n_token
        self.n_layer = n_layer
        self.n_head = n_head
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass of CLIP text encoder.
        
        Args:
            tokens (torch.LongTensor): Tokenized text of shape (batch_size, seq_len)
                                     Token IDs should be in range [0, n_vocab-1]
            
        Returns:
            torch.FloatTensor: Text embeddings of shape (batch_size, seq_len, n_embd)
                              These embeddings can be used for cross-attention conditioning
        """
        # Ensure input is of correct type
        tokens = tokens.type(torch.long)
        
        # === Step 1: Convert tokens to embeddings ===
        # Apply token and positional embeddings
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, n_embd)
        state = self.embedding(tokens)

        # === Step 2: Apply transformer layers ===
        # Process through stack of transformer encoder layers
        for layer in self.layers:
            # Each layer applies self-attention and feedforward processing
            # Shape: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_embd)
            state = layer(state)
            
        # === Step 3: Final normalization ===
        # Apply final layer normalization for stability
        # Shape: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_embd)
        output = self.layernorm(state)
        
        return output
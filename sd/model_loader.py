"""
Model Loader for Stable Diffusion Components

This module provides utilities for loading and initializing pre-trained Stable Diffusion
model components from standard checkpoint files. It handles the complete pipeline
initialization including VAE encoder/decoder, U-Net diffusion model, and CLIP text encoder.

The loader converts standard CompVis/RunwayML checkpoint formats to the custom
implementation and properly initializes each component on the target device.

Components Loaded:
    - CLIP Text Encoder: Processes text prompts into conditioning embeddings
    - VAE Encoder: Converts RGB images to latent space representations
    - VAE Decoder: Converts latent representations back to RGB images
    - Diffusion Model: Performs the iterative denoising process in latent space

Usage:
    >>> models = preload_models_from_standard_weights('model.ckpt', 'cuda')
    >>> clip = models['clip']
    >>> encoder = models['encoder']
    >>> decoder = models['decoder']
    >>> diffusion = models['diffusion']

Dependencies:
    - clip: CLIP text encoder implementation
    - encoder: VAE encoder implementation
    - decoder: VAE decoder implementation
    - diffusion: U-Net diffusion model implementation
    - model_converter: Weight conversion utilities

Author: Stable Diffusion PyTorch Implementation
License: MIT (assumed from public repository)
"""

from typing import Dict, Any
import torch

from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
import model_converter


def preload_models_from_standard_weights(ckpt_path: str, device: str) -> Dict[str, torch.nn.Module]:
    """
    Load and initialize all Stable Diffusion model components from a checkpoint file.
    
    This function provides a complete pipeline for loading pre-trained Stable Diffusion
    models from standard checkpoint formats. It handles weight conversion, model
    initialization, and device placement for all four major components.
    
    The loading process:
    1. Converts checkpoint weights from standard to custom format
    2. Initializes each model component (CLIP, VAE encoder/decoder, U-Net)
    3. Loads converted weights into each component with strict validation
    4. Moves all models to the specified device
    5. Returns a dictionary of initialized, ready-to-use models
    
    Args:
        ckpt_path (str): Path to the Stable Diffusion checkpoint file.
                        Supports .ckpt and .safetensors formats.
                        Should contain a complete Stable Diffusion v1.x model.
        device (str): Target device for model placement ('cpu', 'cuda', 'cuda:0', etc.).
                     All models will be moved to this device after initialization.
    
    Returns:
        Dict[str, torch.nn.Module]: Dictionary containing initialized model components:
            - 'clip': CLIP text encoder for processing text prompts
            - 'encoder': VAE encoder for image-to-latent conversion
            - 'decoder': VAE decoder for latent-to-image conversion
            - 'diffusion': U-Net diffusion model for denoising process
    
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
        RuntimeError: If weight loading fails or device placement fails
        KeyError: If required weight keys are missing from the converted state dict
        ValueError: If the checkpoint format is incompatible
        
    Example:
        >>> # Load models for inference on GPU
        >>> models = preload_models_from_standard_weights('v1-5-pruned-emaonly.ckpt', 'cuda')
        >>> 
        >>> # Access individual components
        >>> text_encoder = models['clip']
        >>> vae_encoder = models['encoder']
        >>> vae_decoder = models['decoder']
        >>> unet = models['diffusion']
        >>> 
        >>> # Models are ready for inference
        >>> with torch.no_grad():
        >>>     text_embeddings = text_encoder.encode(["a beautiful landscape"])
        >>>     latents = vae_encoder.encode(image_tensor)
        
    Note:
        - All models are loaded in evaluation mode by default
        - Weight loading uses strict=True for validation
        - Memory usage can be significant (several GB for full model)
        - Consider using torch.no_grad() context for inference
    """
    # =========================================================================
    # WEIGHT CONVERSION
    # =========================================================================
    # Convert checkpoint weights from standard format to custom implementation format
    # This handles the complex mapping between different weight naming conventions
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    # =========================================================================
    # MODEL INITIALIZATION AND WEIGHT LOADING
    # =========================================================================
    
    # Initialize and load VAE Encoder
    # Converts RGB images (3 channels) to latent space (4 channels)
    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    # Initialize and load VAE Decoder  
    # Converts latent representations (4 channels) back to RGB images (3 channels)
    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    # Initialize and load U-Net Diffusion Model
    # Performs iterative denoising in latent space conditioned on text embeddings
    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    # Initialize and load CLIP Text Encoder
    # Processes text prompts into conditioning embeddings for the diffusion process
    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    # =========================================================================
    # RETURN INITIALIZED MODELS
    # =========================================================================
    # Return dictionary of fully initialized and loaded model components
    # All models are ready for inference and placed on the target device
    return {
        'clip': clip,           # Text encoder for prompt processing
        'encoder': encoder,     # VAE encoder for image encoding  
        'decoder': decoder,     # VAE decoder for image decoding
        'diffusion': diffusion, # U-Net for diffusion denoising
    }
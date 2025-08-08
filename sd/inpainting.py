"""
Stable Diffusion Inpainting Pipeline

This module implements the inpainting functionality for Stable Diffusion, allowing
users to selectively edit portions of images based on text prompts and mask regions.
Inpainting combines the original image with generated content in masked areas.

Key Features:
    - Mask-based selective image editing
    - Text-guided inpainting with prompts
    - Preservation of unmasked areas
    - Smooth blending between original and generated content
    - Support for various mask formats and sizes
    - Configurable inpainting strength and guidance

Inpainting Process:
    1. Encode original image and mask to latent space
    2. Create masked latent representation
    3. Generate noise in masked areas only
    4. Perform denoising guided by text prompt
    5. Blend generated content with original in latent space
    6. Decode final result to RGB image

Dependencies:
    - torch: PyTorch deep learning framework
    - numpy: Numerical computations
    - PIL: Image processing and manipulation
    - pipeline: Core Stable Diffusion pipeline utilities
    - ddpm: DDPM sampling implementation

Author: Stable Diffusion PyTorch Implementation
License: MIT
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Any, Union
from PIL import Image

from pipeline import rescale, get_time_embedding


def prepare_mask(mask: Union[Image.Image, np.ndarray], width: int = 512, height: int = 512) -> torch.Tensor:
    """
    Prepare mask for inpainting by resizing and converting to proper format.
    
    Args:
        mask: Input mask as PIL Image or numpy array. White areas (255) will be inpainted,
              black areas (0) will be preserved.
        width: Target width for mask resizing
        height: Target height for mask resizing
    
    Returns:
        torch.Tensor: Processed mask tensor with shape (1, 1, height, width)
                     Values are 0 (preserve) or 1 (inpaint)
    """
    # Convert PIL Image to numpy array if needed and resize
    if isinstance(mask, Image.Image):
        # Ensure mask is in grayscale and resize
        if mask.mode != 'L':
            mask = mask.convert('L')
        mask = mask.resize((width, height), Image.LANCZOS)
        mask = np.array(mask)
    elif isinstance(mask, np.ndarray):
        # Resize numpy array using PIL if needed
        if mask.shape[:2] != (height, width):
            mask_image = Image.fromarray(mask.astype(np.uint8))
            if len(mask.shape) == 3 and mask.shape[2] > 1:
                mask_image = mask_image.convert('L')
            mask = np.array(mask_image.resize((width, height), Image.LANCZOS))
    
    # Convert to binary mask (0 or 1)
    # White pixels (>127) become 1 (inpaint), black pixels (<=127) become 0 (preserve)
    mask = (mask > 127).astype(np.float32)
    
    # Handle different input formats
    if len(mask.shape) == 3:
        # Convert RGB/RGBA to grayscale by taking first channel
        mask = mask[:, :, 0]
    
    # Convert to tensor and add batch/channel dimensions
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    
    return mask


def prepare_masked_image(image: Image.Image, mask: torch.Tensor) -> torch.Tensor:
    """
    Prepare the original image with masked areas set to a neutral value.
    
    Args:
        image: Original PIL image to be inpainted
        mask: Binary mask tensor (1 = inpaint, 0 = preserve)
    
    Returns:
        torch.Tensor: Masked image tensor with shape (1, 3, H, W)
    """
    # Ensure image is the expected size (should already be resized)
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.LANCZOS)
    
    # Convert image to tensor
    image_tensor = np.array(image)
    image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
    
    # Normalize to [-1, 1] range
    image_tensor = rescale(image_tensor, (0, 255), (-1, 1))
    
    # Rearrange dimensions: (H, W, C) -> (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    
    # Ensure mask has the same spatial dimensions as image
    if mask.shape[-2:] != image_tensor.shape[-2:]:
        # Resize mask to match image dimensions
        mask = torch.nn.functional.interpolate(
            mask.float(), 
            size=image_tensor.shape[-2:], 
            mode='nearest'
        )
    
    # Ensure both tensors are on the same device
    # Move image tensor to the same device as mask
    image_tensor = image_tensor.to(mask.device)
    
    # Apply mask: set masked areas to neutral gray value (0.0 in [-1, 1] range)
    # Expand mask to match image channels: (1, 1, H, W) -> (1, 3, H, W)
    mask_expanded = mask.expand(-1, 3, -1, -1)
    image_tensor = image_tensor * (1 - mask_expanded) + 0.0 * mask_expanded
    
    return image_tensor


def inpaint(
    prompt: str,
    image: Image.Image,
    mask: Union[Image.Image, np.ndarray],
    uncond_prompt: Optional[str] = "",
    strength: float = 0.8,
    do_cfg: bool = True,
    cfg_scale: float = 7.5,
    sampler_name: str = "ddpm",
    n_inference_steps: int = 50,
    models: Dict[str, torch.nn.Module] = {},
    seed: Optional[int] = None,
    device: Optional[str] = None,
    idle_device: Optional[str] = None,
    tokenizer: Any = None,
) -> np.ndarray:
    """
    Perform inpainting on an image using Stable Diffusion.
    
    This function selectively edits portions of an image based on a mask and text prompt.
    Areas marked in white on the mask will be regenerated according to the prompt,
    while black areas will be preserved from the original image.
    
    Args:
        prompt (str): Text description for the content to generate in masked areas.
                     Example: "a red rose in a garden"
        image (PIL.Image): Original image to be inpainted. Will be resized to 512x512.
        mask (Union[PIL.Image, np.ndarray]): Mask defining areas to inpaint.
                                           White pixels = inpaint, black pixels = preserve.
        uncond_prompt (Optional[str]): Negative prompt for classifier-free guidance.
                                     Describes what to avoid. Default: ""
        strength (float): Denoising strength (0.0-1.0). Higher = more change from original.
                         Default: 0.8
        do_cfg (bool): Enable classifier-free guidance for better prompt adherence.
                      Default: True
        cfg_scale (float): Guidance scale (1.0-20.0). Higher = stronger prompt adherence.
                          Default: 7.5
        sampler_name (str): Sampling algorithm. Currently supports "ddpm". Default: "ddpm"
        n_inference_steps (int): Number of denoising steps. More = higher quality.
                               Default: 50
        models (Dict[str, torch.nn.Module]): Dictionary of model components
        seed (Optional[int]): Random seed for reproducible results
        device (Optional[str]): Primary compute device ('cuda', 'cpu', etc.)
        idle_device (Optional[str]): Device for storing idle models
        tokenizer: Text tokenizer for prompt processing
    
    Returns:
        np.ndarray: Inpainted image as numpy array (H, W, 3) with values [0, 255]
    
    Example:
        >>> # Inpaint a flower in a garden scene
        >>> result = inpaint(
        ...     prompt="a beautiful red rose",
        ...     image=original_image,
        ...     mask=mask_image,
        ...     models=loaded_models,
        ...     device="cuda"
        ... )
    """
    with torch.no_grad():
        # =====================================================================
        # INPUT VALIDATION AND SETUP
        # =====================================================================
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        # Setup device management
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # =====================================================================
        # MASK AND IMAGE PREPARATION
        # =====================================================================
        
        # Ensure image is resized to standard dimensions (512x512)
        image = image.resize((512, 512), Image.LANCZOS)
        
        # Prepare mask for inpainting (also resize to 512x512)
        mask_tensor = prepare_mask(mask, 512, 512).to(device)
        
        # Prepare masked image
        masked_image_tensor = prepare_masked_image(image, mask_tensor).to(device)
        
        # Downscale mask to latent space (8x downsampling)
        # Use max pooling to ensure masked areas remain masked after downsampling
        latent_mask = torch.nn.functional.max_pool2d(mask_tensor, kernel_size=8, stride=8)

        # =====================================================================
        # RANDOM NUMBER GENERATOR
        # =====================================================================
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # =====================================================================
        # TEXT ENCODING
        # =====================================================================
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Encode conditional prompt
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)
            
            # Encode unconditional prompt
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            
            # Concatenate for batch processing
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)

        to_idle(clip)

        # =====================================================================
        # LATENT SPACE PREPARATION
        # =====================================================================
        
        # Encode original image to latent space
        encoder = models["encoder"]
        encoder.to(device)
        
        # Encode the original image
        encoder_noise = torch.randn((1, 4, 64, 64), generator=generator, device=device)
        image_latents = encoder(masked_image_tensor, encoder_noise)
        
        # Encode the masked image for reference
        masked_image_latents = encoder(masked_image_tensor, encoder_noise)
        
        to_idle(encoder)

        # =====================================================================
        # SAMPLING SETUP
        # =====================================================================
        if sampler_name == "ddpm":
            from sampler import DDPMSampler
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        elif sampler_name == "ddim":
            from sampler import DDIMSampler
            sampler = DDIMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        elif sampler_name == "euler":
            from sampler import EulerSampler
            sampler = EulerSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}. Supported values: 'ddpm', 'ddim', 'euler'")

        # Set strength for inpainting
        sampler.set_strength(strength=strength)

        # Create initial noisy latents
        latents_shape = (1, 4, 64, 64)  # Latent space dimensions
        noise = torch.randn(latents_shape, generator=generator, device=device)
        
        # Start with encoded image and add noise based on strength
        latents = sampler.add_noise(image_latents, sampler.timesteps[0])

        # =====================================================================
        # DENOISING LOOP WITH MASK APPLICATION
        # =====================================================================
        diffusion = models["diffusion"]
        diffusion.to(device)
        decoder = models["decoder"]
        decoder.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # Time embedding
            time_embedding = get_time_embedding(timestep).to(device)

            # Prepare model input
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            # U-Net prediction
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Perform one denoising step
            latents = sampler.step(timestep, latents, model_output)

            # INPAINTING KEY STEP: Blend original and generated latents based on mask
            # This preserves the original image in unmasked areas
            # latent_mask: 1 = inpaint (use generated), 0 = preserve (use original)
            
            # Add noise to original latents at current timestep for consistency
            noise_timestep = sampler.timesteps[i] if i < len(sampler.timesteps) else sampler.timesteps[-1]
            noisy_original_latents = sampler.add_noise(image_latents, noise_timestep)
            
            # Blend: use generated latents where mask=1, original latents where mask=0
            latents = latents * latent_mask + noisy_original_latents * (1 - latent_mask)

        to_idle(diffusion)

        # =====================================================================
        # FINAL DECODING
        # =====================================================================
        
        # Decode final latents to image
        images = decoder(latents)
        to_idle(decoder)

        # Post-process
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]


def create_circular_mask(width: int = 512, height: int = 512, center: tuple = None, radius: int = 100) -> np.ndarray:
    """
    Create a circular mask for inpainting.
    
    Args:
        width: Mask width
        height: Mask height
        center: Circle center as (x, y). If None, uses image center
        radius: Circle radius in pixels
    
    Returns:
        np.ndarray: Binary mask with circular white region
    """
    if center is None:
        center = (width // 2, height // 2)
    
    # Create coordinate grids
    x, y = np.ogrid[:height, :width]
    
    # Calculate distance from center
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    
    # Convert to 0-255 range
    mask = mask.astype(np.uint8) * 255
    
    return mask


def create_rectangular_mask(width: int = 512, height: int = 512, 
                          x1: int = 100, y1: int = 100, 
                          x2: int = 400, y2: int = 400) -> np.ndarray:
    """
    Create a rectangular mask for inpainting.
    
    Args:
        width: Mask width
        height: Mask height
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates
    
    Returns:
        np.ndarray: Binary mask with rectangular white region
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask

"""
Stable Diffusion Inference Pipeline

This module implements the complete inference pipeline for Stable Diffusion, supporting
both text-to-image and image-to-image generation. It orchestrates the interaction between
all model components (CLIP, VAE, U-Net) to generate high-quality images from text prompts.

Key Features:
    - Text-to-image generation from prompts
    - Image-to-image generation with controllable strength
    - Classifier-free guidance (CFG) for improved quality
    - Multiple sampling algorithms (DDPM)
    - Configurable inference steps and guidance scale
    - Memory-efficient device management
    - Real-time preview during generation

Pipeline Flow:
    1. Text encoding with CLIP
    2. Latent space initialization (random or from input image)
    3. Iterative denoising with U-Net
    4. Latent to image decoding with VAE decoder

Dependencies:
    - torch: PyTorch deep learning framework
    - numpy: Numerical computations
    - tqdm: Progress bars for sampling loops
    - ddpm: DDPM sampling implementation
    - matplotlib: Image visualization during generation

Author: Stable Diffusion PyTorch Implementation
License: MIT (assumed from public repository)
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Any, Union
from PIL import Image

from ddpm import DDPMSampler
import matplotlib.pyplot as plt

# ============================================================================= 
# GLOBAL CONFIGURATION
# =============================================================================
# Standard Stable Diffusion v1.5 dimensions
WIDTH = 512                    # Generated image width in pixels
HEIGHT = 512                   # Generated image height in pixels  
LATENTS_WIDTH = WIDTH // 8     # Latent space width (VAE downsamples by 8x)
LATENTS_HEIGHT = HEIGHT // 8   # Latent space height (VAE downsamples by 8x)

def generate(
    prompt: str,
    uncond_prompt: Optional[str] = None,
    input_image: Optional[Image.Image] = None,
    mask_image: Optional[Union[Image.Image, np.ndarray]] = None,
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
    cancel_flag = None
) -> np.ndarray:
    """
    Generate images using Stable Diffusion with text prompts.
    
    This is the main inference function that orchestrates the complete Stable Diffusion
    pipeline. It supports text-to-image, image-to-image, and inpainting generation with
    configurable parameters for quality, creativity, and performance.
    
    The generation process:
    1. Encode text prompt(s) using CLIP text encoder
    2. Initialize latent noise (random, from input image, or with inpainting mask)
    3. Iteratively denoise latents using U-Net with DDPM sampling
    4. Apply inpainting blending if mask is provided
    5. Decode final latents to RGB image using VAE decoder
    6. Apply classifier-free guidance for improved prompt adherence
    
    Args:
        prompt (str): Main text prompt describing the desired image.
                     Example: "a beautiful landscape with mountains and lakes"
        uncond_prompt (Optional[str]): Negative prompt for classifier-free guidance.
                                     Used to specify what to avoid in the image.
                                     Defaults to empty string if do_cfg=True.
        input_image (Optional[PIL.Image]): Input image for image-to-image or inpainting.
                                         If provided with mask_image, enables inpainting.
                                         If provided without mask, enables img2img.
        mask_image (Optional[Union[PIL.Image, np.ndarray]]): Mask for inpainting.
                                         White areas = inpaint, black areas = preserve.
                                         Only used when input_image is also provided.
        strength (float): Denoising strength for image-to-image generation (0.0-1.0).
                         Higher values = more deviation from input image.
                         Lower values = closer to input image. Default: 0.8
        do_cfg (bool): Enable classifier-free guidance for better prompt adherence.
                      Requires both conditional and unconditional text encodings.
                      Default: True
        cfg_scale (float): Classifier-free guidance scale (1.0-20.0).
                          Higher values = stronger prompt adherence.
                          Lower values = more creative freedom. Default: 7.5
        sampler_name (str): Sampling algorithm to use. Currently supports "ddpm".
                           Default: "ddpm"
        n_inference_steps (int): Number of denoising steps (1-1000).
                               More steps = higher quality but slower generation.
                               Default: 50
        models (Dict[str, torch.nn.Module]): Dictionary containing model components:
                                           - 'clip': CLIP text encoder
                                           - 'encoder': VAE encoder  
                                           - 'decoder': VAE decoder
                                           - 'diffusion': U-Net diffusion model
        seed (Optional[int]): Random seed for reproducible generation.
                            If None, uses random seed. Default: None
        device (Optional[str]): Primary compute device ('cuda', 'cpu', etc.).
                              Where models will be loaded for inference.
        idle_device (Optional[str]): Device for storing idle models to save memory.
                                   Models not currently in use will be moved here.
        tokenizer: Text tokenizer for converting prompts to token IDs.
                  Should be compatible with CLIP tokenization.
    
    Returns:
        np.ndarray: Generated image as numpy array with shape (height, width, 3).
                   Values are in range [0, 255] as uint8.
    
    Raises:
        ValueError: If strength is not between 0 and 1
        ValueError: If sampler_name is not supported
        RuntimeError: If model loading or inference fails
        
    Example:
        >>> # Text-to-image generation
        >>> image = generate(
        ...     prompt="a serene mountain landscape at sunset",
        ...     uncond_prompt="blurry, low quality",
        ...     models=loaded_models,
        ...     device="cuda",
        ...     seed=42
        ... )
        
        >>> # Image-to-image generation  
        >>> modified_image = generate(
        ...     prompt="same landscape but in winter with snow",
        ...     input_image=input_pil_image,
        ...     strength=0.7,
        ...     models=loaded_models,
        ...     device="cuda"
        ... )
        
        >>> # Inpainting generation
        >>> inpainted_image = generate(
        ...     prompt="a beautiful flower garden",
        ...     input_image=original_image,
        ...     mask_image=mask_array,
        ...     strength=0.8,
        ...     models=loaded_models,
        ...     device="cuda"
        ... )
    
    Note:
        - Generation quality increases with more inference steps but takes longer
        - CFG scale 7.5 provides good balance between prompt adherence and creativity
        - For image-to-image, strength 0.8 provides good balance between similarity and change
        - For inpainting, white mask areas are regenerated, black areas are preserved
        - Uses approximately 4-6GB GPU memory during generation
        - Real-time preview images are displayed during generation using matplotlib
    """
    with torch.no_grad():
        # =====================================================================
        # INPUT VALIDATION AND SETUP
        # =====================================================================
        
        # Validate strength parameter for image-to-image generation
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        
        # Check for inpainting mode
        is_inpainting = input_image is not None and mask_image is not None
        
        # Import inpainting functionality if needed
        if is_inpainting:
            from inpainting import inpaint
            return inpaint(
                prompt=prompt,
                image=input_image,
                mask=mask_image,
                uncond_prompt=uncond_prompt,
                strength=strength,
                do_cfg=do_cfg,
                cfg_scale=cfg_scale,
                sampler_name=sampler_name,
                n_inference_steps=n_inference_steps,
                models=models,
                seed=seed,
                device=device,
                idle_device=idle_device,
                tokenizer=tokenizer,
            )

        # Setup device management for memory efficiency
        # Models not in use can be moved to idle_device to save GPU memory
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # =====================================================================
        # RANDOM NUMBER GENERATOR INITIALIZATION
        # =====================================================================
        
        # Initialize random number generator for reproducible generation
        # Ensures consistent results when using the same seed
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()  # Use random seed
        else:
            generator.manual_seed(seed)  # Use specified seed for reproducibility

        # =====================================================================
        # TEXT ENCODING WITH CLIP
        # =====================================================================
        
        # Load CLIP text encoder and move to active device
        clip = models["clip"]
        clip.to(device)

        # Load VAE decoder for final image generation
        decoder = models["decoder"]
        decoder.to(device)
        
        if do_cfg:
            # CLASSIFIER-FREE GUIDANCE: Encode both conditional and unconditional prompts
            # This allows the model to better follow the prompt by contrasting with empty prompt
            
            # Encode the main prompt (conditional)
            # Convert text to tokens with padding to sequence length 77 (CLIP standard)
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # Convert to tensor: (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # Generate text embeddings: (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            
            # Encode the negative/unconditional prompt
            # This represents what we DON'T want in the image
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # Convert to tensor: (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # Generate unconditional embeddings: (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            
            # Concatenate both contexts for batch processing during sampling
            # Shape: (2 * Batch_Size, Seq_Len, Dim) - first half conditional, second half unconditional
            context = torch.cat([cond_context, uncond_context])
        else:
            # STANDARD TEXT ENCODING: Only encode the main prompt
            # Convert text to tokens with padding to sequence length 77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # Convert to tensor: (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # Generate text embeddings: (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
            
        # Move CLIP model to idle device to free GPU memory
        to_idle(clip)

        # =====================================================================
        # SAMPLING ALGORITHM INITIALIZATION
        # =====================================================================
        
        # Initialize the denoising sampler with specified algorithm
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        # Define latent space dimensions for VAE
        # Latents are 1/8 resolution of final image with 4 channels
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # =====================================================================
        # LATENT SPACE INITIALIZATION
        # =====================================================================
        
        if input_image:
            # IMAGE-TO-IMAGE GENERATION: Encode input image to latent space
            encoder = models["encoder"]
            encoder.to(device)

            # Prepare input image for encoding
            # Resize to standard Stable Diffusion dimensions
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # Convert PIL image to numpy array: (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # Convert to PyTorch tensor
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # Normalize from [0, 255] to [-1, 1] range expected by VAE
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # Add batch dimension: (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # Rearrange to PyTorch format: (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Encode image to latent space with added noise for stochasticity
            # Generate random noise for VAE encoding process
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # Encode RGB image to 4-channel latent representation
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to latents based on specified strength
            # Higher strength = more noise = more deviation from input image
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            # Move encoder to idle device to free GPU memory
            to_idle(encoder)
        else:
            # TEXT-TO-IMAGE GENERATION: Start with pure random noise
            # Generate random latent tensor that will be iteratively denoised
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # =====================================================================
        # ITERATIVE DENOISING PROCESS
        # =====================================================================
        
        # Load U-Net diffusion model for denoising
        diffusion = models["diffusion"]
        diffusion.to(device)

        # Main denoising loop with progress bar
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            if cancel_flag is not None and cancel_flag.is_set():
                return None
            # Generate time embedding for current timestep
            # U-Net needs to know what noise level it's working with
            time_embedding = get_time_embedding(timestep).to(device)

            # Prepare model input (current noisy latents)
            model_input = latents

            if do_cfg:
                # CLASSIFIER-FREE GUIDANCE: Process both conditional and unconditional
                # Duplicate latents to process both conditions in parallel
                # Shape: (Batch_Size, 4, H, W) -> (2 * Batch_Size, 4, H, W)
                model_input = model_input.repeat(2, 1, 1, 1)

            # U-Net prediction: Predict the noise that should be removed
            # This is the core of the diffusion process - predicting what noise to subtract
            # Input: noisy latents, text context, timestep
            # Output: predicted noise to remove
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # Apply classifier-free guidance
                # Split predictions: first half is conditional, second half is unconditional
                output_cond, output_uncond = model_output.chunk(2)
                # Blend predictions based on guidance scale
                # Higher cfg_scale = stronger prompt adherence
                # Formula: cfg_scale * (conditional - unconditional) + unconditional
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Update latents by removing predicted noise
            # This is one step of the reverse diffusion process
            latents = sampler.step(timestep, latents, model_output)

            # REAL-TIME PREVIEW: Generate and display intermediate image
            # This allows monitoring the generation progress
            copied_latents = latents.clone()
            images = decoder(copied_latents)
            # Convert from [-1, 1] to [0, 255] range
            images = rescale(images, (-1, 1), (0, 255), clamp=True)
            # Rearrange dimensions for display: (Batch, Channel, H, W) -> (Batch, H, W, Channel)
            images = images.permute(0, 2, 3, 1)
            # Convert to numpy for matplotlib display
            images = images.to("cpu", torch.uint8).numpy()

            # Display current generation state
            plt.imshow(images[0])
            plt.axis('off')
            plt.show()

        # Move diffusion model to idle device after processing
        to_idle(diffusion)

        # =====================================================================
        # FINAL IMAGE GENERATION
        # =====================================================================
        
        # Generate final high-resolution image from denoised latents
        # VAE decoder converts 4-channel latents back to 3-channel RGB image
        # Shape: (Batch_Size, 4, Latents_H, Latents_W) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        
        # Move decoder to idle device to free memory
        to_idle(decoder)

        # Post-process the generated image
        # Convert from model output range [-1, 1] to display range [0, 255]
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # Rearrange tensor dimensions for numpy/PIL compatibility
        # PyTorch format: (Batch, Channel, Height, Width)
        # Display format: (Batch, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        # Convert to numpy array with uint8 values for image saving/display
        images = images.to("cpu", torch.uint8).numpy()
        
        # Return the first (and only) image in the batch
        return images[0]
    
def rescale(
    x: torch.Tensor, 
    old_range: tuple, 
    new_range: tuple, 
    clamp: bool = False
) -> torch.Tensor:
    """
    Rescale tensor values from one range to another.
    
    This utility function linearly transforms tensor values from an old range
    to a new range. Commonly used for converting between model output ranges
    and display/input ranges in image processing pipelines.
    
    Args:
        x (torch.Tensor): Input tensor to rescale
        old_range (tuple): Source range as (min, max) tuple
        new_range (tuple): Target range as (min, max) tuple  
        clamp (bool): Whether to clamp output values to new_range bounds.
                     Prevents values outside target range. Default: False
    
    Returns:
        torch.Tensor: Rescaled tensor with values mapped to new_range
        
    Example:
        >>> # Convert model output [-1, 1] to image pixels [0, 255]
        >>> pixels = rescale(model_output, (-1, 1), (0, 255), clamp=True)
        
        >>> # Convert image pixels [0, 255] to model input [-1, 1]  
        >>> normalized = rescale(image_tensor, (0, 255), (-1, 1))
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    
    # Linear transformation: shift and scale
    x -= old_min                                    # Shift to start at 0
    x *= (new_max - new_min) / (old_max - old_min)  # Scale to new range width
    x += new_min                                    # Shift to new minimum
    
    # Optionally clamp values to prevent overflow/underflow
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep: int) -> torch.Tensor:
    """
    Generate sinusoidal time embeddings for diffusion timesteps.
    
    Creates positional encodings that allow the U-Net to understand what
    noise level it's working with during the denoising process. Uses
    sinusoidal embeddings similar to transformer positional encodings.
    
    The embedding combines multiple frequency components to create a
    unique representation for each timestep that the model can learn to
    interpret and use for appropriate denoising.
    
    Args:
        timestep (int): Current diffusion timestep (0 to max_timesteps).
                       Higher values represent more noise, lower values
                       represent less noise.
    
    Returns:
        torch.Tensor: Time embedding tensor with shape (1, 320).
                     Contains sinusoidal features encoding the timestep.
                     
    Example:
        >>> # Get embedding for timestep 500 (moderate noise)
        >>> embedding = get_time_embedding(500)
        >>> print(embedding.shape)  # torch.Size([1, 320])
        
    Note:
        - Uses 160 frequency components, doubled by sin/cos to get 320 dimensions
        - Frequency range spans from high to low to capture different scales
        - This is a standard approach in diffusion models for time conditioning
    """
    # Generate frequency components with exponential decay
    # Creates 160 frequencies from high (10000^0) to low (10000^-1)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    
    # Create timestep tensor and broadcast with frequencies
    # Shape: (1, 160) after broadcasting
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    
    # Apply sinusoidal encoding: concatenate cos and sin components
    # This creates a unique, smooth embedding for each timestep
    # Final shape: (1, 320) = (1, 160) cos + (1, 160) sin
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

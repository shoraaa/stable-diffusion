"""
DDIM (Denoising Diffusion Implicit Models) Sampler for Stable Diffusion.

This module implements the DDIM sampling algorithm used in Stable Diffusion for
generating images through an accelerated deterministic reverse diffusion process.
DDIM is a more efficient sampler that can achieve good results with fewer steps.

Reference: "Denoising Diffusion Implicit Models" (Song et al., 2020)
https://arxiv.org/pdf/2010.02502.pdf

The DDIM process works by:
1. Starting with pure noise
2. Using a deterministic non-Markovian process to denoise
3. Taking larger steps with adjustable "discretization" via the eta parameter
4. Allowing faster generation with fewer inference steps
"""

import torch
import numpy as np
from typing import Optional, Union


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) sampler.
    
    This class implements the accelerated deterministic sampling process used to generate
    images from noise. It allows for significantly fewer sampling steps compared to DDPM
    while maintaining image quality.
    
    Args:
        generator (torch.Generator): Random number generator for reproducible sampling
        num_training_steps (int): Number of diffusion steps used during training. Defaults to 1000.
        beta_start (float): Starting value of noise schedule. Defaults to 0.00085.
        beta_end (float): Ending value of noise schedule. Defaults to 0.0120.
        eta (float): Controls stochasticity of the sampler (0 = deterministic DDIM, 1 = DDPM)
    """

    def __init__(
        self, 
        generator: torch.Generator, 
        num_training_steps: int = 1000, 
        beta_start: float = 0.00085, 
        beta_end: float = 0.0120,
        eta: float = 0.0
    ):
        # === Noise Schedule Setup ===
        # Create quadratic noise schedule as used in Stable Diffusion
        self.betas = torch.linspace(
            beta_start ** 0.5, 
            beta_end ** 0.5, 
            num_training_steps, 
            dtype=torch.float32
        ) ** 2
        
        # Calculate alpha values: α_t = 1 - β_t
        self.alphas = 1.0 - self.betas
        
        # Calculate cumulative product of alphas: ᾱ_t = ∏(α_s) for s=1 to t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Store values needed for diffusion process
        self.final_alpha_cumprod = self.alphas_cumprod[0]
        
        # Calculate sqrt values for convenience with numerical stability
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod.clamp(min=1e-8))
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt((1.0 - self.alphas_cumprod).clamp(min=1e-8))
        
        # For convenience store one tensor
        self.one = torch.tensor(1.0)
        
        # Set up timestep sequence
        self.num_training_steps = num_training_steps
        self.generator = generator
        
        # Default to deterministic DDIM, but allow adjustment toward DDPM
        self.eta = eta  # 0 = deterministic DDIM, 1 = DDPM
        print(f"Initializing DDIM sampler with eta={eta} (0=deterministic, 1=stochastic)")
        
        # Inference properties (set later)
        self.timesteps = torch.ones((1,), dtype=torch.long)
        
        # For img2img
        self.start_step = 0

    def set_inference_timesteps(self, n_inference_steps: int = 50):
        """
        Set the sequence of timesteps to use for the inference/sampling process.
        
        Args:
            n_inference_steps (int): The number of diffusion steps to perform
        """
        # Make sure n_inference_steps is within a reasonable range
        if n_inference_steps <= 0:
            raise ValueError(f"Number of inference steps must be positive, got {n_inference_steps}")
        if n_inference_steps > self.num_training_steps:
            print(f"Warning: n_inference_steps ({n_inference_steps}) > num_training_steps ({self.num_training_steps})")
            n_inference_steps = self.num_training_steps
        
        # Evenly space out the inference timesteps across the full training range
        # For DDIM, we can use far fewer steps than the training timesteps
        timesteps = torch.linspace(
            self.num_training_steps - 1, 0, n_inference_steps, dtype=torch.long
        )
        
        # Store timesteps for sampling process
        self.timesteps = timesteps
        
        # Create mapping between inference timesteps and original training timesteps
        step_ratio = self.num_training_steps // n_inference_steps
        self.timestep_map = {}
        for i, t in enumerate(timesteps):
            self.timestep_map[i] = t
            
        # Store alphas and sigmas for inference steps
        self.ddim_alphas = self.alphas_cumprod[timesteps]
        # For the previous alpha value, we need the alpha for the previous step
        self.ddim_alpha_prev = torch.cat([self.alphas_cumprod[0:1], self.alphas_cumprod[timesteps[:-1]]])
        
        # Calculate sigma values for adding noise (if eta > 0) with improved numerical stability
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-8
        
        # Handle potential division by zero with epsilon
        alpha_ratio = torch.div(
            self.ddim_alpha_prev + epsilon, 
            self.ddim_alphas + epsilon
        )
        
        # Variance calculation with numerical safeguards 
        one_minus_alpha_ratio = torch.clamp(1.0 - alpha_ratio, min=0.0, max=1.0)
        
        # Calculate standard deviation (sigma)
        # Formula: σ_t = η * √[(1 - α_{t-1})/(1 - α_t) * (1 - α_t/α_{t-1})]
        variance = torch.div(
            1.0 - self.ddim_alpha_prev + epsilon, 
            1.0 - self.ddim_alphas + epsilon
        ) * one_minus_alpha_ratio
        
        # Ensure variance is positive before sqrt
        variance = torch.clamp(variance, min=0.0)
        
        # Calculate final sigma with eta parameter
        self.ddim_sigma = self.eta * torch.sqrt(variance)
        
        # Print debug info about sigma values
        print(f"DDIM sigma stats: min={self.ddim_sigma.min().item():.6f}, max={self.ddim_sigma.max().item():.6f}")
            
    def step(self, timestep: torch.Tensor, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        """
        Perform one DDIM sampling step.
        
        Args:
            timestep (torch.Tensor): Current diffusion timestep
            latents (torch.Tensor): Current noisy latents
            model_output (torch.Tensor): Predicted noise from diffusion model
            
        Returns:
            torch.Tensor: Denoised latents for next step
        """
        try:
            # Find the step index in our inference sequence
            step_index = (self.timesteps == timestep).nonzero().item()
            
            # Debug information for tensor placement
            device_info = f"Timestep device: {timestep.device}, Latents device: {latents.device}, Model output device: {model_output.device}"
            
            # Get alpha values for current step and ensure they're on the same device as latents
            alpha = self.ddim_alphas[step_index].to(latents.device)
            alpha_prev = self.ddim_alpha_prev[step_index].to(latents.device)
            sigma = self.ddim_sigma[step_index].to(latents.device)
            
            # Debug pixel value ranges
            latent_stats = f"Latents: min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}"
            noise_stats = f"Pred noise: min={model_output.min().item():.4f}, max={model_output.max().item():.4f}, mean={model_output.mean().item():.4f}"
            
            # The model predicts epsilon (noise)
            pred_noise = model_output
            
            # Calculate the predicted original sample x_0
            sqrt_alpha = torch.sqrt(alpha + 1e-8)  # Add epsilon to prevent numerical issues
            sqrt_one_minus_alpha = torch.sqrt((1 - alpha).clamp(min=1e-8))  # Clamp to prevent negative values
            
            # Make sure we're not dividing by zero and handle potential numerical instability
            # x_0 = (x_t - √(1-α) * ε) / √α
            pred_original_sample = (latents - sqrt_one_minus_alpha * pred_noise) / (sqrt_alpha)
            
            # Apply a conservative clipping to prevent extreme values
            pred_original_sample = torch.clamp(pred_original_sample, -10.0, 10.0)
            
            # Direction pointing to x_t
            # Formula: √(1-α_prev-σ²) * ε
            # Ensure we don't have negative values under the sqrt by clamping
            dir_coef = (1 - alpha_prev - sigma**2).clamp(min=1e-8)
            dir_xt = torch.sqrt(dir_coef) * pred_noise
            
            # Compute predicted previous sample using the DDIM formula
            # x_{t-1} = √α_{t-1} * x_0 + √(1-α_{t-1}-σ²) * ε_t + σ * z
            sqrt_alpha_prev = torch.sqrt(alpha_prev + 1e-8)  # Add epsilon for stability
            prev_latents = sqrt_alpha_prev * pred_original_sample + dir_xt
            
            # Add randomness if eta > 0
            if self.eta > 0:
                # Make sure we're generating noise on the correct device
                noise = torch.randn(
                    latents.shape, 
                    generator=self.generator, 
                    device=latents.device
                )
                # Add noise scaled by sigma
                prev_latents = prev_latents + sigma * noise
            
            # Debug intermediate results
            pred_orig_stats = f"Pred original: min={pred_original_sample.min().item():.4f}, max={pred_original_sample.max().item():.4f}"
            result_stats = f"Result: min={prev_latents.min().item():.4f}, max={prev_latents.max().item():.4f}"
            
            # Debug: Check for NaN/Inf values
            if torch.isnan(prev_latents).any() or torch.isinf(prev_latents).any():
                print(f"WARNING: NaN or Inf detected in DDIM step output at timestep {timestep}")
                print(f"Debug info: {device_info}")
                print(f"Stats before fix: {latent_stats}, {noise_stats}, {pred_orig_stats}, {result_stats}")
                # Replace NaN/Inf values with zeros to avoid propagation
                prev_latents = torch.nan_to_num(prev_latents, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Apply final clipping for stability
            prev_latents = torch.clamp(prev_latents, -10.0, 10.0)
            
            return prev_latents
            
        except Exception as e:
            print(f"Error in DDIM step: {e}")
            # In case of error, don't change the latents (return as is)
            return latents
    
    def set_strength(self, strength: float = 1.0) -> None:
        """
        Set the denoising strength for image-to-image generation.
        
        This method is used in img2img pipelines where we start from an existing
        image rather than pure noise. The strength parameter controls how much
        of the original image structure is preserved.
        
        Args:
            strength (float): Denoising strength between 0.0 and 1.0
                            - 1.0: Start from pure noise (like txt2img)
                            - 0.8: Add significant noise, major changes
                            - 0.5: Moderate changes to original image
                            - 0.2: Minor changes, preserve most structure
                            - 0.0: No changes (return original image)
        """
        # Calculate how many initial denoising steps to skip
        # Higher strength = fewer skipped steps = more denoising = more changes
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        
        # Skip the initial timesteps (start from partially noised image)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
        
        print(f"Set DDIM strength to {strength}, starting from step {start_step}/{self.num_inference_steps}")
    
    def add_noise(
        self, 
        latents: torch.Tensor, 
        noise: torch.Tensor = None, 
        timestep: int = None
    ) -> torch.Tensor:
        """
        Add noise to the latent vectors according to the diffusion schedule.
        
        Args:
            latents (torch.Tensor): Clean latent representation (x_0)
            noise (torch.Tensor, optional): Random noise to add (epsilon). If None, generates new noise.
            timestep (int, optional): Timestep determining noise level. If None, uses first timestep.
            
        Returns:
            torch.Tensor: Noisy latents
        """
        try:
            # Default to first timestep if not specified
            if timestep is None:
                timestep = self.timesteps[0]
                
            # Generate random noise if not provided
            if noise is None:
                noise = torch.randn(
                    latents.shape,
                    generator=self.generator, 
                    device=latents.device
                )
                
            # Check for valid timestep range
            if timestep >= self.num_training_steps:
                raise ValueError(f"Timestep {timestep} is out of bounds for {self.num_training_steps} steps")
                
            # Get scaling factors for the clean image and the noise
            # Ensure they're on the correct device
            sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timestep].to(latents.device)
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timestep].to(latents.device)
            
            # Debug input stats
            latent_stats = f"Latents: min={latents.min().item():.4f}, max={latents.max().item():.4f}"
            noise_stats = f"Noise: min={noise.min().item():.4f}, max={noise.max().item():.4f}"
            
            # Apply the noise using the forward diffusion formula:
            # x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
            noisy_latents = sqrt_alpha_cumprod * latents + sqrt_one_minus_alpha_cumprod * noise
            
            # Debug output stats
            output_stats = f"Output: min={noisy_latents.min().item():.4f}, max={noisy_latents.max().item():.4f}"
            
            # Debug: Check for NaN/Inf values
            if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                print(f"WARNING: NaN or Inf detected in add_noise output at timestep {timestep}")
                print(f"Debug info: {latent_stats}, {noise_stats}, {output_stats}")
                # Replace NaN/Inf values with zeros to avoid propagation
                noisy_latents = torch.nan_to_num(noisy_latents, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Apply conservative clipping to prevent extreme values
            noisy_latents = torch.clamp(noisy_latents, -10.0, 10.0)
            
            return noisy_latents
            
        except Exception as e:
            print(f"Error in DDIM add_noise: {e}")
            # In case of error, return original latents
            return latents

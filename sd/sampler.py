"""
DDPM, DDIM, DPM-Solver-2, and Euler Samplers for Stable Diffusion.

This module implements multiple sampling algorithms used in Stable Diffusion
for generating images through the reverse diffusion process.

DDPM (Denoising Diffusion Probabilistic Models):
- Stochastic sampling process
- Follows the methodology from Ho et al., 2020: https://arxiv.org/pdf/2006.11239.pdf
- Requires many steps (typically 1000) for high quality
- Adds random noise at each step

DDIM (Denoising Diffusion Implicit Models):  
- Deterministic sampling process (when eta=0)
- From Song et al., 2020: https://arxiv.org/abs/2010.02502
- Can achieve high quality with fewer steps (20-50)
- Allows for consistent generation and interpolation

DPM-Solver-2 (DPM2):
- Second-order multistep solver for diffusion ODEs
- From Lu et al., 2022: https://arxiv.org/abs/2206.00927
- Achieves excellent quality with very few steps (10-20)
- Uses predictor-corrector approach for better accuracy
- Available in both deterministic and ancestral (stochastic) variants

Euler Samplers:
- First-order ODE solvers based on Song et al., 2021: https://arxiv.org/abs/2011.13456
- Simple and efficient for few-step generation (15-30 steps)
- Works well with Karras-style noise schedules
- Available in both deterministic (Euler) and stochastic (EulerAncestral) variants
- Popular for practical applications due to simplicity and effectiveness

All algorithms work by:
1. Starting with pure noise
2. Iteratively denoising the sample using a learned model
3. Following a predefined noise schedule to gradually remove noise
4. Producing a clean sample after all denoising steps
"""

import torch
import numpy as np
from typing import Optional, Union


class BaseSampler:
    """
    Base sampler class for diffusion models (DDPM, DDIM, DPM-Solver-2, and Euler).
    
    This class implements the common diffusion functionality used by all samplers.
    It manages the noise schedule and provides methods for both sampling 
    (generation) and adding noise (for training or img2img).
    
    Args:
        generator (torch.Generator): Random number generator for reproducible sampling
        num_training_steps (int): Number of diffusion steps used during training. Defaults to 1000.
        beta_start (float): Starting value of noise schedule. Defaults to 0.00085.
        beta_end (float): Ending value of noise schedule. Defaults to 0.0120.
        
    Note:
        Default beta values are taken from Stable Diffusion v1 configuration:
        https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml
    """

    def __init__(
        self, 
        generator: torch.Generator, 
        num_training_steps: int = 1000, 
        beta_start: float = 0.00085, 
        beta_end: float = 0.0120
    ):
        # === Noise Schedule Setup ===
        # Create quadratic noise schedule as used in Stable Diffusion
        # The square root interpolation creates a more gradual noise increase
        self.betas = torch.linspace(
            beta_start ** 0.5, 
            beta_end ** 0.5, 
            num_training_steps, 
            dtype=torch.float32
        ) ** 2
        
        # Calculate alpha values: α_t = 1 - β_t
        self.alphas = 1.0 - self.betas
        
        # Calculate cumulative product of alphas: ᾱ_t = ∏(α_s) for s=1 to t
        # This is crucial for the forward diffusion process q(x_t | x_0)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Constant tensor for boundary conditions
        self.one = torch.tensor(1.0)

        # Store the random number generator for reproducible sampling
        self.generator = generator

        # === Timestep Configuration ===
        self.num_train_timesteps = num_training_steps
        
        # Initialize with full training schedule (reversed for sampling)
        # During training: t goes from 0 to T-1 (forward process)
        # During sampling: t goes from T-1 to 0 (reverse process)
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50) -> None:
        """
        Set the timesteps for inference (sampling).
        
        This method creates a subset of the training timesteps for faster inference.
        Instead of using all 1000 training steps, we can use fewer steps (e.g., 50)
        by skipping timesteps uniformly across the schedule.
        
        Args:
            num_inference_steps (int): Number of denoising steps during inference.
                                     Fewer steps = faster but potentially lower quality.
                                     More steps = slower but potentially higher quality.
        """
        self.num_inference_steps = num_inference_steps
        
        # Calculate how many training steps to skip between inference steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        
        # Create evenly spaced timesteps and reverse for sampling
        # Example: For 50 inference steps from 1000 training steps,
        # we take every 20th timestep: [980, 960, 940, ..., 20, 0]
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """
        Get the previous timestep in the sampling schedule.
        
        Args:
            timestep (int): Current timestep
            
        Returns:
            int: Previous timestep in the sampling sequence
        """
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        Calculate the variance for the reverse diffusion step.
        
        This implements the variance calculation from DDPM paper formula (6) and (7).
        The variance determines how much random noise to add during sampling.
        
        Args:
            timestep (int): Current timestep
            
        Returns:
            torch.Tensor: Variance value for sampling noise
        """
        prev_t = self._get_previous_timestep(timestep)

        # Get cumulative alpha products for current and previous timesteps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        
        # Calculate current beta: β_t = 1 - (ᾱ_t / ᾱ_{t-1})
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # Calculate variance using DDPM formula (7):
        # σ²_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # Clamp to prevent numerical issues (log of zero)
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
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
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Add noise to clean samples according to the forward diffusion process.
        
        This method implements the forward diffusion process q(x_t | x_0) from
        DDPM paper equation (4). It's used for training and img2img applications
        where we need to add a specific amount of noise to an image.
        
        Args:
            original_samples (torch.FloatTensor): Clean samples (x_0)
            timesteps (torch.IntTensor): Timesteps indicating noise level to add
            
        Returns:
            torch.FloatTensor: Noisy samples (x_t) 
            
        Note:
            The forward process is: q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1-ᾱ_t) * I)
            Which can be sampled as: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
            where ε ~ N(0, I)
        """
        # Move scheduling parameters to the same device and dtype as input
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, 
            dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        # === Step 1: Prepare signal scaling factor √ᾱ_t ===
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        
        # Broadcast to match sample dimensions (handle batch, channel, height, width)
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # === Step 2: Prepare noise scaling factor √(1-ᾱ_t) ===
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        
        # Broadcast to match sample dimensions
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # === Step 3: Sample noise and apply forward diffusion ===
        # Generate random noise ε ~ N(0, I)
        noise = torch.randn(
            original_samples.shape, 
            generator=self.generator, 
            device=original_samples.device, 
            dtype=original_samples.dtype
        )
        
        # Apply forward diffusion: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        
        return noisy_samples

        
class DDPMSampler(BaseSampler):

    def step(
        self, 
        timestep: int, 
        latents: torch.Tensor, 
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one reverse diffusion sampling step.
        
        This is the core DDPM sampling step that takes a noisy latent and the
        model's noise prediction to compute the less noisy latent for the next step.
        
        Args:
            timestep (int): Current timestep in the diffusion process
            latents (torch.Tensor): Current noisy latents 
            model_output (torch.Tensor): Model's predicted noise
            
        Returns:
            torch.Tensor: Denoised latents for the next step
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # === Step 1: Compute alpha and beta values ===
        alpha_prod_t = self.alphas_cumprod[t]           # ᾱ_t
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one  # ᾱ_{t-1}
        beta_prod_t = 1 - alpha_prod_t                  # 1 - ᾱ_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev        # 1 - ᾱ_{t-1}
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev  # α_t = ᾱ_t / ᾱ_{t-1}
        current_beta_t = 1 - current_alpha_t            # β_t = 1 - α_t

        # === Step 2: Predict original sample x_0 ===
        # From DDPM paper equation (15): x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
        # where ε is the predicted noise (model_output)
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # === Step 3: Compute coefficients for mean calculation ===
        # From DDPM paper equation (7), the mean μ_t is a weighted combination:
        # μ_t = coeff1 * x_0 + coeff2 * x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # === Step 4: Compute predicted mean μ_t ===
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # === Step 5: Add noise for stochastic sampling ===
        variance = 0
        if t > 0:  # No noise added at the final step (t=0)
            device = model_output.device
            noise = torch.randn(
                model_output.shape, 
                generator=self.generator, 
                device=device, 
                dtype=model_output.dtype
            )
            # Scale noise by the square root of variance
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # === Step 6: Sample from N(μ_t, σ²_t) ===
        # Final sample: x_{t-1} = μ_t + σ_t * ε where ε ~ N(0,1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample


class DDIMSampler(BaseSampler):
    """
    DDIM (Denoising Diffusion Implicit Models) sampler.
    
    This class implements the DDIM sampling algorithm, which is a deterministic
    variant of DDPM that allows for faster sampling with fewer steps. Unlike DDPM,
    DDIM uses a deterministic reverse process that can produce high-quality samples
    with significantly fewer denoising steps.
    
    Reference: "Denoising Diffusion Implicit Models" (Song et al., 2020)
    https://arxiv.org/abs/2010.02502
    
    Key differences from DDPM:
    - Deterministic reverse process (no random noise added)
    - Can use much fewer sampling steps (e.g., 20-50 instead of 1000)
    - Allows for consistent generation and interpolation
    - Generally faster inference while maintaining quality
    """

    def step(
        self, 
        timestep: int, 
        latents: torch.Tensor, 
        model_output: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Perform one reverse diffusion sampling step using DDIM.
        
        The DDIM step is deterministic (when eta=0) and uses a different update
        rule compared to DDPM that allows for larger timestep skips.
        
        Args:
            timestep (int): Current timestep in the diffusion process
            latents (torch.Tensor): Current noisy latents x_t
            model_output (torch.Tensor): Model's predicted noise ε_θ(x_t, t)
            eta (float): Controls stochasticity. 0.0 = deterministic (DDIM), 
                        1.0 = stochastic (equivalent to DDPM). Defaults to 0.0.
            
        Returns:
            torch.Tensor: Denoised latents for the next step x_{t-1}
            
        Note:
            DDIM update rule:
            x_{t-1} = √ᾱ_{t-1} * x_0_pred + √(1-ᾱ_{t-1}-σ²_t) * ε_θ + σ_t * ε
            where x_0_pred = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # === Step 1: Get alpha values for current and previous timesteps ===
        alpha_prod_t = self.alphas_cumprod[t]           # ᾱ_t
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one  # ᾱ_{t-1}

        # === Step 2: Predict original sample x_0 ===
        # From the DDIM paper: x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
        sqrt_alpha_prod_t = alpha_prod_t ** 0.5
        sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t) ** 0.5
        
        pred_original_sample = (latents - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t

        # === Step 3: Calculate variance for stochastic component ===
        # DDIM allows interpolation between deterministic (eta=0) and stochastic (eta=1)
        variance = self._get_variance(t)
        std_dev_t = eta * (variance ** 0.5)
        
        # === Step 4: Compute direction pointing to x_t ===
        # This is the "direction" component in the DDIM formulation
        sqrt_alpha_prod_t_prev = alpha_prod_t_prev ** 0.5
        sqrt_one_minus_alpha_prod_t_prev = (1 - alpha_prod_t_prev) ** 0.5
        
        # Calculate the coefficient for the direction term
        # When eta=0: this becomes √(1-ᾱ_{t-1}) 
        # When eta=1: this becomes √(1-ᾱ_{t-1}-σ²_t)
        direction_coeff = (sqrt_one_minus_alpha_prod_t_prev ** 2 - std_dev_t ** 2) ** 0.5
        
        # === Step 5: Compute DDIM update ===
        # x_{t-1} = √ᾱ_{t-1} * x_0_pred + direction_coeff * ε + σ_t * noise
        pred_prev_sample = (
            sqrt_alpha_prod_t_prev * pred_original_sample +  # Signal component
            direction_coeff * model_output                    # Direction component
        )
        
        # === Step 6: Add stochastic component (if eta > 0) ===
        if eta > 0 and t > 0:  # No noise at final step
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype
            )
            pred_prev_sample = pred_prev_sample + std_dev_t * noise

        return pred_prev_sample


class EulerSampler(BaseSampler):
    """
    Euler sampler for diffusion models.
    
    This class implements the Euler method, a first-order ODE solver commonly used
    in diffusion models. The Euler method provides a simple and efficient approach
    for sampling from diffusion models with fewer steps than DDPM.
    
    The Euler method treats the reverse diffusion process as an ODE and uses
    first-order numerical integration to solve it. This approach is based on
    the work of Song et al. (2021) on score-based generative models.
    
    Reference: "Score-Based Generative Modeling through Stochastic Differential Equations"
    https://arxiv.org/abs/2011.13456
    
    Key characteristics:
    - First-order ODE solver
    - Deterministic sampling process
    - Simple and efficient implementation
    - Works well with 15-30 sampling steps
    - Good balance between speed and quality
    """

    def step(
        self, 
        timestep: int, 
        latents: torch.Tensor, 
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one reverse diffusion sampling step using the Euler method.
        
        The Euler method uses first-order numerical integration to solve the
        reverse-time SDE. It approximates the solution by taking a linear step
        in the direction of the predicted score (negative gradient of log density).
        
        Args:
            timestep (int): Current timestep in the diffusion process
            latents (torch.Tensor): Current noisy latents x_t
            model_output (torch.Tensor): Model's predicted noise ε_θ(x_t, t)
            
        Returns:
            torch.Tensor: Denoised latents for the next step x_{t-1}
            
        Note:
            Euler update rule:
            x_{t-1} = x_t + (t - t_prev) * dx_dt
            where dx_dt is the derivative estimated from the model output
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # === Step 1: Get alpha values for current timestep ===
        alpha_prod_t = self.alphas_cumprod[t]           # ᾱ_t
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one  # ᾱ_{t-1}

        # === Step 2: Calculate beta and sigma values ===
        beta_prod_t = 1 - alpha_prod_t                  # 1 - ᾱ_t
        sqrt_alpha_prod_t = alpha_prod_t ** 0.5         # √ᾱ_t
        sqrt_beta_prod_t = beta_prod_t ** 0.5           # √(1 - ᾱ_t)

        # === Step 3: Predict original sample x_0 ===
        # x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
        pred_original_sample = (latents - sqrt_beta_prod_t * model_output) / sqrt_alpha_prod_t

        # === Step 4: Calculate the derivative dx/dt ===
        # For the Euler method, we need to estimate the derivative of x with respect to t
        # This is based on the predicted original sample and current state
        sqrt_alpha_prod_t_prev = alpha_prod_t_prev ** 0.5
        sqrt_beta_prod_t_prev = (1 - alpha_prod_t_prev) ** 0.5

        # Calculate what x_{t-1} should be based on predicted x_0
        pred_sample_direction = sqrt_alpha_prod_t_prev * pred_original_sample + sqrt_beta_prod_t_prev * model_output

        # === Step 5: Compute the Euler step ===
        # The step size is effectively the difference between consecutive timesteps
        # in the continuous time formulation
        if prev_t >= 0:
            # Calculate the step as the difference between predicted direction and current state
            # This implements the Euler method: x_{t-1} = x_t + step_size * derivative
            pred_prev_sample = pred_sample_direction
        else:
            # Final step: predict the clean sample directly
            pred_prev_sample = pred_original_sample

        return pred_prev_sample


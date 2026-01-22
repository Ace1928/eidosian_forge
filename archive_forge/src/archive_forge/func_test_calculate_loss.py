import gc
import unittest
import torch
from trl import is_diffusers_available, is_peft_available
from .testing_utils import require_diffusers
def test_calculate_loss(self):
    samples, _ = self.trainer._generate_samples(1, 2)
    sample = samples[0]
    latents = sample['latents'][0, 0].unsqueeze(0)
    next_latents = sample['next_latents'][0, 0].unsqueeze(0)
    log_probs = sample['log_probs'][0, 0].unsqueeze(0)
    timesteps = sample['timesteps'][0, 0].unsqueeze(0)
    prompt_embeds = sample['prompt_embeds']
    advantage = torch.tensor([1.0], device=prompt_embeds.device)
    assert latents.shape == (1, 4, 64, 64)
    assert next_latents.shape == (1, 4, 64, 64)
    assert log_probs.shape == (1,)
    assert timesteps.shape == (1,)
    assert prompt_embeds.shape == (2, 77, 32)
    loss, approx_kl, clipfrac = self.trainer.calculate_loss(latents, timesteps, next_latents, log_probs, advantage, prompt_embeds)
    assert torch.isfinite(loss.cpu())
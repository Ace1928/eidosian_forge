import os
import warnings
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Tuple
from warnings import warn
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import whoami
from ..models import DDPOStableDiffusionPipeline
from . import BaseTrainer, DDPOConfig
from .utils import PerPromptStatTracker
def compute_rewards(self, prompt_image_pairs, is_async=False):
    if not is_async:
        rewards = []
        for images, prompts, prompt_metadata in prompt_image_pairs:
            reward, reward_metadata = self.reward_fn(images, prompts, prompt_metadata)
            rewards.append((torch.as_tensor(reward, device=self.accelerator.device), reward_metadata))
    else:
        rewards = self.executor.map(lambda x: self.reward_fn(*x), prompt_image_pairs)
        rewards = [(torch.as_tensor(reward.result(), device=self.accelerator.device), reward_metadata.result()) for reward, reward_metadata in rewards]
    return zip(*rewards)
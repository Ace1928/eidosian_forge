import inspect
import math
import os
import time
import typing
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Union
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object, is_deepspeed_available
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (
from ..core import (
from ..import_utils import is_npu_available, is_torch_greater_2_0, is_xpu_available
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper, create_reference_model
from . import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig, RunningMoments
from transformers import pipeline
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
def record_step_stats(self, kl_coef: float, **data):
    """
        Record training step statistics.


        Args:
            kl_coef (`float`):
                KL coefficient
            data (`dict`):
                Dictionary of training step data

        Returns:
            stats (`dict`):
                Dictionary of training step statistics
        """
    mask = data.pop('masks')
    kls = data.pop('kls')
    kl_list = (kls * mask).sum(axis=-1)
    mean_kl = kl_list.mean()
    mean_entropy = (-data['logprobs'] * mask).sum(axis=-1).mean()
    mean_non_score_reward = masked_mean(data['non_score_reward'], mask)
    mean_scores = data['scores'].mean()
    std_scores = data['scores'].std()
    if mean_kl.item() < -1.0:
        warnings.warn(f'KL divergence is starting to become negative: {mean_kl.item():.2f} - this might be a precursor for failed training. sometimes this happens because the generation kwargs are not correctly set. Please make sure that the generation kwargs are set correctly, or review your training hyperparameters.')
    stats = {'objective/kl': mean_kl, 'objective/kl_dist': kl_list, 'objective/logprobs': data['logprobs'], 'objective/ref_logprobs': data['ref_logprobs'], 'objective/kl_coef': kl_coef, 'objective/entropy': mean_entropy, 'ppo/mean_non_score_reward': mean_non_score_reward, 'ppo/mean_scores': mean_scores, 'ppo/std_scores': std_scores}
    query_lens = torch.tensor([len(query) for query in data['queries']], dtype=torch.float)
    response_lens = torch.tensor([len(response) for response in data['responses']], dtype=torch.float)
    stats['tokens/queries_len_mean'] = torch.mean(query_lens).cpu().numpy().item()
    stats['tokens/queries_len_std'] = torch.std(query_lens).cpu().numpy().item()
    stats['tokens/queries_dist'] = query_lens.cpu().numpy()
    stats['tokens/responses_len_mean'] = torch.mean(response_lens).cpu().numpy().item()
    stats['tokens/responses_len_std'] = torch.std(response_lens).cpu().numpy().item()
    stats['tokens/responses_dist'] = response_lens.cpu().numpy()
    for k, v in data['train_stats'].items():
        stats[f'ppo/{k}'] = torch.mean(v, axis=0)
    stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
    return stats
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
def log_stats(self, stats: dict, batch: dict, rewards: List[torch.FloatTensor], columns_to_log: typing.Iterable[str]=('query', 'response')):
    """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
        """
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards).to(self.current_device)
    rewards = self.accelerator.gather(rewards).flatten()
    if self.config.log_with == 'wandb':
        import wandb
        if any((column_to_log not in batch.keys() for column_to_log in columns_to_log)):
            raise ValueError(f'Columns to log {columns_to_log} are not present in the batch {batch.keys()}.')
        batch_list = [batch[column_to_log] for column_to_log in columns_to_log]
        if self.is_distributed:
            gathered_batch_list = []
            for b in batch_list:
                flattened = gather_object(b)
                gathered_batch_list.append(flattened)
            batch_list = gathered_batch_list
    if self.accelerator.is_main_process:
        logs = {}
        if 'query' not in batch.keys() and 'response' not in batch.keys():
            warnings.warn("The game logs will not be logged because the batch does not contain the keys 'query' and 'response'. ")
        elif self.config.log_with == 'wandb':
            table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
            logs.update({'game_log': wandb.Table(columns=[*columns_to_log, 'reward'], rows=table_rows)})
        logs.update(stats)
        for k, v in logs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                logs[k] = v.float()
        logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy().item()
        logs['env/reward_std'] = torch.std(rewards).cpu().numpy().item()
        logs['env/reward_dist'] = rewards.cpu().numpy()
        if self.config.log_with == 'tensorboard':
            self.current_step += 1
        self.accelerator.log(logs, step=self.current_step if self.config.log_with == 'tensorboard' else None)
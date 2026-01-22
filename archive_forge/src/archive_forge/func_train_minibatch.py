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
@PPODecorators.empty_device_cache()
def train_minibatch(self, old_logprobs: torch.FloatTensor, values: torch.FloatTensor, logprobs: torch.FloatTensor, logits: torch.FloatTensor, vpreds: torch.FloatTensor, mask: torch.LongTensor, advantages: torch.FloatTensor, returns: torch.FloatTensor):
    """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [mini_batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [mini_batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [mini_batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [mini_batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [mini_batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """
    self.model.train()
    loss_p, loss_v, train_stats = self.loss(old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns)
    loss = loss_p + loss_v
    self.accelerator.backward(loss)
    if self.config.max_grad_norm is not None:
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
    self.optimizer.step()
    self.optimizer.zero_grad()
    return train_stats
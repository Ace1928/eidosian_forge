from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.llama.configuration_llama import LlamaConfig
@torch.jit.script
def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    return (weights * torch.nn.functional.cross_entropy(logits, labels, reduction='none')).sum()
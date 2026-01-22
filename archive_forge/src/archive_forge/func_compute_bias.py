import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wavlm import WavLMConfig
def compute_bias(self, query_length: int, key_length: int) -> torch.FloatTensor:
    context_position = torch.arange(query_length, dtype=torch.long)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position
    relative_position_bucket = self._relative_positions_bucket(relative_position)
    relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
    values = self.rel_attn_embed(relative_position_bucket)
    values = values.permute([2, 0, 1])
    return values
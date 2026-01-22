import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, LayerNorm
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_fsmt import FSMTConfig
def make_weight(self, num_positions, embedding_dim, padding_idx):
    weight = self.get_embedding(num_positions, embedding_dim, padding_idx)
    if not hasattr(self, 'weight'):
        super().__init__(num_positions, embedding_dim, padding_idx, _weight=weight)
    else:
        weight = weight.to(dtype=self.weight.dtype, device=self.weight.device)
        self.weight = nn.Parameter(weight)
    self.weight.detach_()
    self.weight.requires_grad = False
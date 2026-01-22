import copy
import math
import warnings
from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_longt5 import LongT5Config
def compute_side_bias(self, mask: torch.Tensor, global_segment_ids: torch.Tensor) -> torch.Tensor:
    side_attention_mask = torch.eq(mask[..., None], global_segment_ids[:, None, :])[:, None, ...]
    attention_side_bias = torch.where(side_attention_mask > 0, 0.0, -10000000000.0)
    side_relative_position = _make_side_relative_position_ids(mask, self.global_block_size)
    side_relative_position_bucket = self._relative_position_bucket(side_relative_position, bidirectional=not self.is_decoder, num_buckets=self.relative_attention_num_buckets, max_distance=self.relative_attention_max_distance)
    side_bias = self.global_relative_attention_bias(side_relative_position_bucket)
    side_bias = side_bias.permute([0, 3, 1, 2])
    attention_side_bias = attention_side_bias + side_bias
    return attention_side_bias
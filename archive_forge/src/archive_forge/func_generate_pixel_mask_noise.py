import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
def generate_pixel_mask_noise(pixel_values, pixel_mask=None, mask_ratio=0.75):
    """Generate noise for audio masking."""
    batch_size, seq_len = pixel_values.shape[:2]
    noise = torch.rand((batch_size, seq_len), device=pixel_values.device)
    len_keep = int(seq_len * (1 - mask_ratio))
    return (noise, len_keep)
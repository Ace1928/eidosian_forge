import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_owlv2 import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig
def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:
    box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
    box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)
    box_coord_bias = torch.log(box_coordinates + 0.0001) - torch.log1p(-box_coordinates + 0.0001)
    box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
    box_size_bias = torch.log(box_size + 0.0001) - torch.log1p(-box_size + 0.0001)
    box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
    return box_bias
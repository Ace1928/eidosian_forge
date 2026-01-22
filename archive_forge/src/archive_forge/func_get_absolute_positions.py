import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitdet import VitDetConfig
def get_absolute_positions(self, abs_pos_embeddings, has_cls_token, height, width):
    """
        Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token dimension for the
        original embeddings.

        Args:
            abs_pos_embeddings (`torch.Tensor`):
                Absolute positional embeddings with (1, num_position, num_channels).
            has_cls_token (`bool`):
                If true, has 1 embedding in abs_pos_embeddings for cls token.
            height (`int`):
                Height of input image tokens.
            width (`int`):
                Width of input image tokens.

        Returns:
            Absolute positional embeddings after processing with shape (1, height, width, num_channels)
        """
    if has_cls_token:
        abs_pos_embeddings = abs_pos_embeddings[:, 1:]
    num_position = abs_pos_embeddings.shape[1]
    size = int(math.sqrt(num_position))
    if size * size != num_position:
        raise ValueError('Absolute position embeddings must be a square number.')
    if size != height or size != width:
        new_abs_pos_embeddings = nn.functional.interpolate(abs_pos_embeddings.reshape(1, size, size, -1).permute(0, 3, 1, 2), size=(height, width), mode='bicubic', align_corners=False)
        return new_abs_pos_embeddings.permute(0, 2, 3, 1)
    else:
        return abs_pos_embeddings.reshape(1, height, width, -1)
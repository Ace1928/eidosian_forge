from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
def remove_keypoints_from_borders(keypoints: torch.Tensor, scores: torch.Tensor, border: int, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Removes keypoints (and their associated scores) that are too close to the border"""
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < height - border)
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < width - border)
    mask = mask_h & mask_w
    return (keypoints[mask], scores[mask])
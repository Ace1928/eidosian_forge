import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig
def pair_wise_sigmoid_focal_loss(inputs: Tensor, labels: Tensor, alpha: float=0.25, gamma: float=2.0) -> Tensor:
    """
    A pair wise version of the focal loss, see `sigmoid_focal_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha (float, *optional*, defaults to 0.25):
            Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma (float, *optional*, defaults to 2.0):
            Exponent of the modulating factor \\\\(1 - p_t\\\\) to balance easy vs hard examples.

    Returns:
        `torch.Tensor`: The computed loss between each pairs.
    """
    if alpha < 0:
        raise ValueError('alpha must be positive')
    height_and_width = inputs.shape[1]
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    prob = inputs.sigmoid()
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    focal_pos = (1 - prob) ** gamma * cross_entropy_loss_pos
    focal_pos *= alpha
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))
    focal_neg = prob ** gamma * cross_entropy_loss_neg
    focal_neg *= 1 - alpha
    loss = torch.matmul(focal_pos, labels.T) + torch.matmul(focal_neg, (1 - labels).T)
    return loss / height_and_width
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_beit import BeitConfig
def compute_loss(self, logits, auxiliary_logits, labels):
    upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    if auxiliary_logits is not None:
        upsampled_auxiliary_logits = nn.functional.interpolate(auxiliary_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
    main_loss = loss_fct(upsampled_logits, labels)
    loss = main_loss
    if auxiliary_logits is not None:
        auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
        loss += self.config.auxiliary_loss_weight * auxiliary_loss
    return loss
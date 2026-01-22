import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import is_accelerate_available, logging
from ...utils.backbone_utils import load_backbone
from .configuration_mask2former import Mask2FormerConfig
def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
    auxiliary_logits: List[Dict(str, Tensor)] = []
    for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
        auxiliary_logits.append({'masks_queries_logits': aux_binary_masks, 'class_queries_logits': aux_classes})
    return auxiliary_logits
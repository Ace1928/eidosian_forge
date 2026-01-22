import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
    num_masks = sum([len(classes) for classes in class_labels])
    num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=device)
    world_size = 1
    if is_accelerate_available():
        if PartialState._shared_state != {}:
            num_masks = reduce(num_masks)
            world_size = PartialState().num_processes
    num_masks = torch.clamp(num_masks / world_size, min=1)
    return num_masks
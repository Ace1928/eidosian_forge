import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ...utils.backbone_utils import load_backbone
from .configuration_tvp import TvpConfig
def loss_distance(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
    """
        Measure the distance of mid points.
        """
    mid_candidates = torch.div(torch.add(candidates_start_time, candidates_end_time), 2.0)
    mid_groundtruth = torch.div(torch.add(start_time, end_time), 2.0)
    distance_diff = torch.div(torch.max(mid_candidates, mid_groundtruth) - torch.min(mid_candidates, mid_groundtruth), duration).clamp(min=0.2)
    return distance_diff
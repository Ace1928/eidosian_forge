import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, is_torchvision_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone
from .configuration_deta import DetaConfig
def set_low_quality_matches_(self, match_labels, match_quality_matrix):
    """
        Produce additional matches for predictions that have only low-quality matches. Specifically, for each
        ground-truth G find the set of predictions that have maximum overlap with it (including ties); for each
        prediction in that set, if it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of :paper:`Faster R-CNN`.
        """
    highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
    _, pred_inds_with_highest_quality = nonzero_tuple(match_quality_matrix == highest_quality_foreach_gt[:, None])
    match_labels[pred_inds_with_highest_quality] = 1
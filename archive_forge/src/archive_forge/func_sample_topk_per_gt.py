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
def sample_topk_per_gt(pr_inds, gt_inds, iou, k):
    if len(gt_inds) == 0:
        return (pr_inds, gt_inds)
    gt_inds2, counts = gt_inds.unique(return_counts=True)
    scores, pr_inds2 = iou[gt_inds2].topk(k, dim=1)
    gt_inds2 = gt_inds2[:, None].repeat(1, k)
    pr_inds3 = torch.cat([pr[:c] for c, pr in zip(counts, pr_inds2)])
    gt_inds3 = torch.cat([gt[:c] for c, gt in zip(counts, gt_inds2)])
    return (pr_inds3, gt_inds3)
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.fx
import torchvision
from torch import nn, Tensor
from torchvision.ops.boxes import box_area
from ..utils import _log_api_usage_once
from .roi_align import roi_align
def initLevelMapper(k_min: int, k_max: int, canonical_scale: int=224, canonical_level: int=4, eps: float=1e-06):
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)
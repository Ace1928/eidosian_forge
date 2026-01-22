import copy
import logging
from typing import List, Optional
import torch
import torch.nn as nn
from torch._dynamo.utils import detect_fake_mode
from torch._utils_internal import print_graph
from torch.fx.experimental.optimization import (
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights
from .. import config
from ..fx_utils import matches_module_function_pattern
from ..pattern_matcher import (
from ..utils import is_cpu_device
from .group_batch_fusion import group_batch_fusion_passes
from .misc_patterns import numpy_compat_normalization
def transpose_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    if bias is None:
        return torch.matmul(input.transpose(-1, -2), weight.t())
    return torch.matmul(input.transpose(-1, -2), weight.t()) + bias
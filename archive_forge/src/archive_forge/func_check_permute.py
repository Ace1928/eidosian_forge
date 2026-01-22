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
def check_permute(node: torch.fx.Node) -> bool:
    ranks = len(node.meta['tensor_meta'].shape)
    if len(node.args) > 3:
        permutation = [node.args[i] % ranks for i in range(1, ranks + 1)]
    elif 'permutation' in node.kwargs and node.kwargs['permutation'] is not None and (len(node.kwargs['permutation']) > 2):
        permutation = [i % ranks for i in node.kwargs['permutation']]
    else:
        return False
    allowed_permutation = list(range(ranks))
    allowed_permutation[-1] = ranks - 2
    allowed_permutation[-2] = ranks - 1
    return permutation == allowed_permutation
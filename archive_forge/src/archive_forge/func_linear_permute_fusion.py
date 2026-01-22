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
def linear_permute_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.nodes:
        if node.op == 'call_method' and node.target == 'permute' and check_permute(node):
            if len(node.args) > 0:
                input_node = node.args[0]
            else:
                input_node = node.kwargs['input']
            if input_node.op == 'call_function' and input_node.target == torch.nn.functional.linear:
                normalized = NormalizedLinearNode(input_node)
                input = normalized.get_input()
                weight = normalized.get_weight()
                bias = normalized.get_bias()
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(linear_transpose, args=(input, weight, bias))
                    node.replace_all_uses_with(fused_node)
                    module.graph.erase_node(node)
                    if len(input_node.users) == 0:
                        module.graph.erase_node(input_node)
    module.graph.lint()
    module.recompile()
    return module
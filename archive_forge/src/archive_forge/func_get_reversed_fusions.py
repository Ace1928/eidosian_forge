import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization.backend_config import get_native_backend_config
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers
from torch.ao.quantization.utils import getattr_from_fqn
from .ns_types import NSNodeTargetType
from torch.ao.quantization import (
from typing import Dict, Tuple, Set, Callable, Any, Union, List
def get_reversed_fusions() -> List[Tuple[NSFusionType, int]]:
    """
    Set of potential fusions, in reverse order.  The order is reversed
    to match how fusion patterns are defined in quantization code.

    Fusion format:
    ((fusion_op_0, fusion_op_1), base_op_idx)

    Where base_op_idx is the idx of the op we should use to match other related
    ops. Note: base_op_idx is specified in non-reverse order, i.e. a base_op_idx
    of 0 represents the first op in regular (non-reverse) order, 1 represents the
    second op, etc.
    """
    results: List[Tuple[NSFusionType, int]] = []
    all_quant_patterns = _get_pattern_to_quantize_handlers(get_native_backend_config())
    default_base_op_idx = 0
    for quant_pattern in all_quant_patterns.keys():
        if isinstance(quant_pattern, tuple) and len(quant_pattern) == 2 and isinstance(quant_pattern[1], tuple) and (len(quant_pattern[1]) == 2):
            quant_pattern = (quant_pattern[0], quant_pattern[1][0], quant_pattern[1][1])
        if isinstance(quant_pattern, tuple):
            results.append((quant_pattern, default_base_op_idx))
        for cls in (ObserverBase, FakeQuantizeBase):
            if isinstance(quant_pattern, tuple):
                new_pattern = (cls, *quant_pattern)
            else:
                new_pattern = (cls, quant_pattern)
            results.append((new_pattern, default_base_op_idx))
    fp16_em_base_op_idx = 1
    patterns_to_add = [((('to', torch.float16), F.relu, F.linear, 'dequantize'), fp16_em_base_op_idx), ((nn.BatchNorm1d, nn.Conv1d), default_base_op_idx), ((nn.BatchNorm2d, nn.Conv2d), default_base_op_idx), ((nn.BatchNorm3d, nn.Conv3d), default_base_op_idx), ((nn.ReLU, nn.BatchNorm1d, nn.Conv1d), default_base_op_idx), ((nn.ReLU, nn.BatchNorm2d, nn.Conv2d), default_base_op_idx), ((nn.ReLU, nn.BatchNorm3d, nn.Conv3d), default_base_op_idx)]
    for p in patterns_to_add:
        results.append(p)
        results.append(((ObserverBase, *p[0]), p[1]))
        results.append(((FakeQuantizeBase, *p[0]), p[1]))
    return results
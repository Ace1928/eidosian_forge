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
def get_type_a_related_to_b(base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]]) -> Set[Tuple[NSNodeTargetType, NSNodeTargetType]]:
    type_a_related_to_b: Set[Tuple[NSNodeTargetType, NSNodeTargetType]] = set()
    for s in base_name_to_sets_of_related_ops.values():
        s_list = list(s)
        for idx_0 in range(0, len(s_list)):
            for idx_1 in range(idx_0, len(s_list)):
                type_a_related_to_b.add((s_list[idx_0], s_list[idx_1]))
                type_a_related_to_b.add((s_list[idx_1], s_list[idx_0]))
    return type_a_related_to_b
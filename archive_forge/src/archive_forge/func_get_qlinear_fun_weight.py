import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
from torch.fx import GraphModule
from torch.fx.graph import Node
from .utils import (
from .ns_types import (
from typing import List, Optional, Dict, Callable
def get_qlinear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    packed_weight_node = node.args[1]
    assert isinstance(packed_weight_node, Node)
    assert packed_weight_node.op == 'get_attr'
    packed_weight = getattr_from_fqn(gm, packed_weight_node.target)
    (weight, _bias), _name = packed_weight.__getstate__()
    return weight
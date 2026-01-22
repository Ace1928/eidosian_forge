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
def get_conv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    weight_arg_node = node.args[1]
    assert isinstance(weight_arg_node, Node)
    weight_node = return_first_non_observer_node(weight_arg_node, gm)
    assert isinstance(weight_node, Node)
    assert weight_node.op == 'get_attr'
    weight = getattr_from_fqn(gm, weight_node.target)
    return weight.detach()
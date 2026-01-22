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
def get_linear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    linear_second_arg = node.args[1]
    assert isinstance(linear_second_arg, Node)
    if linear_second_arg.op == 'call_module':
        weight_arg_node = node.args[1]
        assert isinstance(weight_arg_node, Node)
        weight_node = weight_arg_node.args[0]
        assert isinstance(weight_node, Node)
        assert weight_node.op == 'get_attr'
        weight = getattr_from_fqn(gm, weight_node.target)
        return weight.detach()
    elif linear_second_arg.op == 'call_method':
        assert linear_second_arg.op == 'call_method'
        dequant_node = node.args[1]
        assert isinstance(dequant_node, Node)
        to_fp16_node = dequant_node.args[0]
        assert isinstance(to_fp16_node, Node)
        target_dtype = to_fp16_node.args[1]
        weight_node = to_fp16_node.args[0]
        assert isinstance(weight_node, Node)
        assert weight_node.op == 'get_attr'
        weight = getattr_from_fqn(gm, weight_node.target)
        return weight.detach().to(target_dtype)
    else:
        assert linear_second_arg.op == 'get_attr'
        weight = getattr_from_fqn(gm, linear_second_arg.target)
        return weight.detach()
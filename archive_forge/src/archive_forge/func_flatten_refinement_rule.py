from functools import reduce
import torch
import operator
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
from typing import Callable, Dict
from torch.fx.node import Target, Node
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.fx.experimental.refinement_types import Equality
import itertools
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
import sympy
@register_refinement_rule(torch.flatten)
def flatten_refinement_rule(n: Node):
    """
    Generates equality constraints between the dimensions of the input and output
    that will not be involved in the flatten operation
    """
    assert isinstance(n.args[0], Node)
    eq_const = []
    start_dim = 1
    end_dim = -1
    if len(n.args) > 1:
        assert isinstance(n.args[1], int)
        start_dim = n.args[1]
    if len(n.args) > 2:
        assert isinstance(n.args[2], int)
        end_dim = n.args[2]
    if isinstance(n.type, TensorType) and isinstance(n.args[0].type, TensorType):
        l = len(n.type.__args__)
        arg_type = n.args[0].type
        start_dim = l if start_dim == -1 else start_dim
        end_dim = l + end_dim + 1 if end_dim < 0 else end_dim + 1
        for t1, t2 in zip(n.type.__args__[0:start_dim], arg_type.__args__[0:start_dim]):
            eq_const.append(Equality(t1, t2))
        for t1, t2 in zip(n.type.__args__[end_dim:], arg_type.__args__[end_dim:]):
            eq_const.append(Equality(t1, t2))
    return eq_const
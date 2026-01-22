import torch
import operator
import warnings
from typing import Callable, Dict, Iterable
from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, CalcProduct, \
from torch.fx.experimental.migrate_gradual_types.operation import \
from torch.fx.node import Target, Node
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar, gen_tvar, \
from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
@register_inference_rule(torch._C._nn.linear)
def torch_linear_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    weight_dims, counter = gen_tensor_dims(2, counter)
    equality_constraint = BinConstraintT(symbols[n.args[1]], TensorType(weight_dims), op_eq)
    constraints, counter = linear_constraints(n, weight_dims[1], weight_dims[0], symbols, counter)
    return ([equality_constraint] + constraints, counter)
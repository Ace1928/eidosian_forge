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
@register_inference_rule('transpose')
def transpose_inference_rule(n: Node, symbols, constraints, counter):
    """
    Can be considered as a sequence of two index selects, so we generate constraints accordingly
    """
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], int)
    assert isinstance(n.args[2], int)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    from_arg = symbols[n.args[0]]
    assert isinstance(from_arg, TVar)
    is_dyn = Conj([BinConstraintT(from_arg, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)])
    c3 = Disj([Transpose(i + 1, from_arg, n.args[1], n.args[2], output) for i in range(MAX_TENSOR_RANK)])
    return ([Disj([is_dyn, c3])], counter)
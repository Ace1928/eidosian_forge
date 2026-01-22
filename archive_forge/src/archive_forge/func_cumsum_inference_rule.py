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
@register_inference_rule(torch.cumsum)
def cumsum_inference_rule(n: Node, symbols, constraints, counter):
    """
    Input and output shapes should be equal
    We should verify that the index is valid
    """
    assert isinstance(n.args[0], Node)
    arg_1 = n.args[1] if len(n.args) > 1 else n.kwargs['dim']
    assert isinstance(arg_1, int)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    input = symbols[n.args[0]]
    input_dyn = BinConstraintT(input, Dyn, op_eq)
    output_dyn = BinConstraintT(output, Dyn, op_eq)
    c1 = Conj([input_dyn, output_dyn])
    c2 = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims, counter = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(new_dims)
        c_tensor_i = Conj([BinConstraintT(input, TensorType(new_dims), op_eq), BinConstraintT(output, TensorType(new_dims), op_eq)] + [range_check(arg_1, i)] + nat_constraints)
        c2.append(c_tensor_i)
    dyn_or_tensor = Disj([c1, Disj(c2)])
    return ([dyn_or_tensor], counter)
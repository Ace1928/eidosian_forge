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
@register_inference_rule('dim')
def torch_dim_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    my_dim, counter = gen_dvar(counter)
    symbols[n] = my_dim
    input = symbols[n.args[0]]
    input_dyn = BinConstraintT(input, Dyn, op_eq)
    output_dyn = BinConstraintD(my_dim, Dyn, op_eq)
    c1 = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)
        c_tensor_i = Conj([BinConstraintT(input, TensorType(new_dims_rhs_1), op_eq), BinConstraintD(my_dim, i, op_eq)])
        c1.append(c_tensor_i)
    return ([Disj([Conj([input_dyn, output_dyn]), Disj(c1)])], counter)
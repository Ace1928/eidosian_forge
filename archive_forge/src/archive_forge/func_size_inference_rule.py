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
@register_inference_rule('size')
def size_inference_rule(n: Node, symbols, constraints, counter):
    """
    The constraint is just lhs = rhs.
    Ex: size = input_ids.size()
    """
    if len(n.args) == 1:
        size, counter = gen_tvar(counter)
        symbols[n] = size
        input = symbols[n.args[0]]
        c = BinConstraintT(input, size, op_eq)
        return ([c], counter)
    elif len(n.args) == 2:
        if isinstance(n.args[1], int):
            size_index, counter = gen_dvar(counter)
            symbols[n] = size_index
            input = symbols[n.args[0]]
            c2 = [GetItem(i + 1, n.args[1], size_index, input) for i in range(MAX_TENSOR_RANK)]
            c3 = BinConstraintD(0, size_index, op_leq)
            input_dyn = BinConstraintT(input, Dyn, op_eq)
            output_dyn = BinConstraintD(size_index, Dyn, op_eq)
            c1 = Conj([input_dyn, output_dyn])
            return ([Disj([c1, Conj([Disj(c2), c3])])], counter)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
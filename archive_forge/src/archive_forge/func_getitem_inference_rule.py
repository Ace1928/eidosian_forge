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
@register_inference_rule(operator.getitem)
def getitem_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    if isinstance(n.args[1], int):
        get_item_output, counter = gen_dvar(counter)
        symbols[n] = get_item_output
        get_item_arg = symbols[n.args[0]]
        assert isinstance(get_item_arg, TVar)
        input_dyn = BinConstraintT(get_item_arg, Dyn, op_eq)
        output_dyn = BinConstraintD(get_item_output, Dyn, op_eq)
        c1 = Conj([input_dyn, output_dyn])
        c2 = [GetItem(i + 1, n.args[1], get_item_output, get_item_arg) for i in range(MAX_TENSOR_RANK)]
        c3 = BinConstraintD(0, get_item_output, op_leq)
        return ([Disj([c1, Conj([Disj(c2), c3])])], counter)
    elif isinstance(n.args[1], tuple):
        get_item_output, counter = gen_tvar(counter)
        symbols[n] = get_item_output
        if n.args[0] in symbols:
            get_item_arg = symbols[n.args[0]]
            assert isinstance(get_item_arg, TVar)
            input_dyn = BinConstraintT(get_item_arg, Dyn, op_eq)
            output_dyn = BinConstraintT(get_item_output, Dyn, op_eq)
            c1 = Conj([input_dyn, output_dyn])
            c2 = [GetItemTensor(i + 1, n.args[1], get_item_output, get_item_arg) for i in range(MAX_TENSOR_RANK)]
        else:
            return ([], counter)
        return ([Disj([c1, *c2])], counter)
    else:
        raise RuntimeError('Method not yet implemented')
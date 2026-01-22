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
@register_inference_rule(operator.eq)
def eq_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], (Node, int))
    assert isinstance(n.args[1], (Node, int))
    e1 = symbols[n.args[0]] if isinstance(n.args[0], Node) else n.args[0]
    e2 = symbols[n.args[1]] if isinstance(n.args[1], Node) else n.args[1]
    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(e1, TVar) and isinstance(e2, TVar):
            eq_tensor, counter = gen_tvar(counter)
            symbols[n] = eq_tensor
            return gen_broadcasting_constraints(e1, e2, symbols, counter, eq_tensor)
        elif isinstance(e1, DVar) and isinstance(e2, DVar):
            eq_constraint = BinConstraintD(e1, e2, op_eq)
            my_eq, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_eq, eq_constraint, op_eq)
            return ([equality_constraint], counter)
        else:
            raise RuntimeError('Sort Mismatch')
    elif isinstance(n.args[0], Node) and (not isinstance(n.args[1], Node)):
        if isinstance(e1, DVar):
            eq_constraint = BinConstraintD(e1, e2, op_eq)
            my_eq, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_eq, eq_constraint, op_eq)
            return ([equality_constraint], counter)
        else:
            raise NotImplementedError('Method not yet implemented')
    else:
        raise NotImplementedError('Method not yet implemented')
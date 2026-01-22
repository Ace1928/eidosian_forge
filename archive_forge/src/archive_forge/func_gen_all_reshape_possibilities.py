import copy
import itertools
from torch.fx.experimental.migrate_gradual_types.constraint_generator import BinConstraintT, MAX_TENSOR_RANK
from torch.fx.experimental.migrate_gradual_types.constraint import T, BinConstraintD, Conj, Constraint, DVar, TVar, \
from torch.fx.experimental.migrate_gradual_types.constraint import Disj, TGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import DGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import CalcConv, CalcMaxPool
from torch.fx.experimental.migrate_gradual_types.constraint import CalcProduct, CanReshape
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, Prod, F, GetItem, GetItemTensor, IndexSelect
from torch.fx.experimental.migrate_gradual_types.operation import op_eq, op_precision, op_leq, op_matching
from torch.fx.experimental.migrate_gradual_types.operation import op_consistency, op_neq
from torch.fx.experimental.migrate_gradual_types.operation import op_mul, op_add, op_sub, op_div, op_mod
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar
from torch.fx.tensor_type import TensorType, Dyn
from typing import Callable, Dict, List
def gen_all_reshape_possibilities(list_of_dims, target):
    """
    Consider all possibilities what the input dimensions could be (number or dynamic)
    Then generate the appropriate constraints using multiplication or mod depending on the possibility
    The possibilities we consider here are the cross product of being equal to dyn or not equal to dyn
    for the input. Target is fixed because at most one dimension could be dyn.
    We have different cases for this.

    Args:
        list_of_dims: The input list of dimensions
        target: The tensor we want to reshape to

    Returns: A disjunction of transformed reshape constraints

    """
    all_possibilities = generate_all_int_dyn_dim_possibilities(list_of_dims)
    all_constraints = []
    for p in all_possibilities:
        to_multiply = []
        p = list(p)
        for constraint in p:
            assert isinstance(constraint, BinConstraintD)
            if constraint.op == op_neq:
                to_multiply.append(constraint.lhs)
        if not to_multiply:
            all_constraints.append(Conj(p))
        elif len(to_multiply) < len(list_of_dims):
            all_constraints.append(Conj(p + [is_target_div_by_dim(target, Prod(to_multiply))]))
        else:
            all_constraints.append(Conj(p + [BinConstraintD(Prod(list_of_dims), Prod(target), op_eq)]))
    return Disj(all_constraints)
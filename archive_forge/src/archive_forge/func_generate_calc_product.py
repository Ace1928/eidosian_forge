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
@register_transformation_rule(CalcProduct)
def generate_calc_product(constraint, counter):
    """
    Transform flatten constraints
    """
    start = constraint.start
    end = constraint.end
    dims = constraint.dims_to_flatten
    flattened = constraint.flattened
    n = len(constraint.dims_to_flatten)
    boundary_check = 0 <= start and start < end and (end <= n)
    c_boundary = T() if boundary_check else F()
    lhs = dims[0:start]
    rhs = dims[end:]
    mid = dims[start:end]
    all_possibilities = generate_all_int_dyn_dim_possibilities(mid)
    all_constraints = []
    for p in all_possibilities:
        p = list(p)
        contains_dyn = not all((constraint.op == op_neq for constraint in p))
        if contains_dyn:
            mid_var = [Dyn]
            total_constraints = lhs + mid_var + rhs
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                all_constraints.append(Conj([BinConstraintT(flattened, TensorType(lhs + mid_var + rhs), op_eq)] + p))
        else:
            new_var, counter = gen_dvar(counter)
            mid_eq_prod = Conj([BinConstraintD(new_var, Prod(mid), op_eq), BinConstraintD(new_var, Dyn, op_neq)])
            mid_var = [new_var]
            total_constraints = lhs + mid_var + rhs
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                all_constraints.append(Conj([BinConstraintT(flattened, TensorType(lhs + mid_var + rhs), op_eq), mid_eq_prod] + p))
    return (Conj([Disj(all_constraints), c_boundary]), counter)
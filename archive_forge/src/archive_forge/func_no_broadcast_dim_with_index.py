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
def no_broadcast_dim_with_index(d1: List[DVar], d2: List[DVar], d3: List[DVar], d4: List[DVar], i: int):
    """
    Args:
        d1: input 1
        d2: input 2
        d3: simulated broadcasting for input 1
        d4: simulated broadcasting for input 2
        i: the rank of the resulting tensor addition

    Returns: Constraints for when no broadcasting occurs
    """
    return Conj([Disj([Conj([BinConstraintD(d1[i], 1, op_eq), BinConstraintD(d2[i], 1, op_eq)]), Conj([BinConstraintD(d1[i], 1, op_neq), BinConstraintD(d2[i], 1, op_neq)])]), BinConstraintD(d1[i], d3[i], op_eq), BinConstraintD(d2[i], d4[i], op_eq)])
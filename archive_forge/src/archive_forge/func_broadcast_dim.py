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
def broadcast_dim(tensor_input1, tensor_input2, res1, res2, index, padding=False):
    """
    Apply broadcasting to the 'index' dimension of tensor_input1.
    Args:
        tensor_input1: should represent [d1, ..., d_index, ...] where d_index = 1
        tensor_input2: represents the second input
        res1: broadcasted result 1
        res2: broadcasted result 2
        index: the index to broadcast
        padding: If padding was used, then tensor_input1[index] does not exist

    Returns:

    """
    if tensor_input1[index] is None:
        assert padding
    if not padding:
        return Conj([BinConstraintD(tensor_input1[index], 1, op_eq), BinConstraintD(res1[index], res2[index], op_eq), BinConstraintD(res2[index], tensor_input2[index], op_eq)])
    else:
        return Conj([BinConstraintD(res1[index], res2[index], op_eq), BinConstraintD(res2[index], tensor_input2[index], op_eq)])
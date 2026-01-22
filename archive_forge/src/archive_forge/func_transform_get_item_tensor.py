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
@register_transformation_rule(GetItemTensor)
def transform_get_item_tensor(constraint, counter):
    """
    When the index is a tuple, then the output will be a tensor
    TODO: we have to check if this is the case for all HF models

    The cases we are covering here are a tuple with one of:
     - slice with default argument
     - None

     None appends 1 to the input tensor dimensions
     so each occurrence of 'None' increases the rank by 1

     slice with default arguments does not change the rank
    """
    assert isinstance(constraint.index_tuple, tuple)
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    nat_constraints = gen_nat_constraints(dims)
    none_c = constraint.index_tuple.count(None)
    resulting_tensor_dims = (none_c + len(dims)) * [None]
    dim_index = 0
    for i in range(len(constraint.index_tuple)):
        if constraint.index_tuple[i] is None:
            resulting_tensor_dims[i] = 1
        elif constraint.index_tuple[i] == slice(None, None, None):
            pass
        else:
            raise NotImplementedError('Method not yet implemented')
    dim_index = 0
    for i in range(len(resulting_tensor_dims)):
        if resulting_tensor_dims[i] is None:
            resulting_tensor_dims[i] = dims[dim_index]
            dim_index += 1
    is_valid_index = valid_index_tensor(constraint.index_tuple, dims)
    if len(resulting_tensor_dims) > 4:
        return (F(), counter)
    else:
        constraints = [BinConstraintT(constraint.input_var, TensorType(dims), op_eq), BinConstraintT(constraint.res, TensorType(resulting_tensor_dims), op_eq), *nat_constraints, is_valid_index]
        return (Conj(constraints), counter)
from torch.fx.experimental.migrate_gradual_types.constraint import TVar, DVar, BinConstraintD, \
from torch.fx.experimental.migrate_gradual_types.operation import op_leq
def gen_nat_constraints(list_of_dims):
    """
    Generate natural number constraints for dimensions
    """
    return [BinConstraintD(0, d, op_leq) for d in list_of_dims]
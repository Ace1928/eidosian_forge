from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
def is_dim(d):
    return isinstance(d, (DVar, int)) or d == Dyn
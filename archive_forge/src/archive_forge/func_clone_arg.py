import torch
from .make_fx import make_fx_check
from .aot_autograd import aot_autograd_check
from .fake_tensor import fake_check
from torch._subclasses.schema_check_mode import SchemaCheckMode
def clone_arg(arg):
    if isinstance(arg, torch.Tensor):
        return arg.clone()
    if isinstance(arg, (tuple, list)):
        return type(arg)((clone_arg(a) for a in arg))
    return arg
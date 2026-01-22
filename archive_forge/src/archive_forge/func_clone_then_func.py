import torch
from .make_fx import make_fx_check
from .aot_autograd import aot_autograd_check
from .fake_tensor import fake_check
from torch._subclasses.schema_check_mode import SchemaCheckMode
def clone_then_func(*args, **kwargs):
    args, kwargs = clone_tensors(args, kwargs)
    return func(*args, **kwargs)
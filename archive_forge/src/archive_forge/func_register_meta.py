import functools
import torch
import torch.library
import torchvision.extension  # noqa: F401
def register_meta(op_name, overload_name='default'):

    def wrapper(fn):
        if torchvision.extension._has_ops():
            get_meta_lib().impl(getattr(getattr(torch.ops.torchvision, op_name), overload_name), fn)
        return fn
    return wrapper
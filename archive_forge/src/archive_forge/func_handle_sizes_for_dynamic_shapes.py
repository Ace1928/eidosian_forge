import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_methods_invocations import wrapper_set_seed
import torch.utils._pytree as pytree
def handle_sizes_for_dynamic_shapes(func, args, kwargs):

    def f(args, kwargs, extra_args, extra_kwargs):
        if extra_args:
            for i, t in extra_args:
                args[i] = t.size()
        if extra_kwargs:
            for k, t in extra_kwargs.items():
                kwargs[k] = t.size()
        return func(*args, **kwargs)
    extra_args = []
    extra_kwargs = {}
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Size):
            extra_args.append((i, torch.empty(arg, device='cpu')))
    for key, value in kwargs.items():
        if isinstance(value, torch.Size):
            extra_kwargs[key] = torch.empty(value, device='cpu')
    return (f, args, kwargs, extra_args, extra_kwargs)
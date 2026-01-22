import dataclasses
import functools
from importlib import import_module
from typing import Any, List, Optional
from functorch.compile import min_cut_rematerialization_partition
import torch
from torch import _guards
from torch._functorch.compilers import ts_compile
from .common import aot_autograd
from .registry import register_debug_backend as register_backend
@register_backend
def pre_dispatch_eager(gm, fake_tensor_inputs):
    from torch.fx.experimental.proxy_tensor import make_fx

    def runnable_gm(*args):
        return torch.fx.Interpreter(gm).run(*args)
    pre_dispatch_gm = make_fx(runnable_gm, pre_dispatch=True)(*fake_tensor_inputs)
    pre_dispatch_gm.print_readable()
    return pre_dispatch_gm
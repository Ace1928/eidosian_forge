import contextlib
from typing import Optional, Union, List, Set, Dict, Any
import warnings
from dataclasses import dataclass
import torch
import torchgen
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
def is_traceable_wrapper_subclass(t):
    """
    Returns whether or not a tensor subclass that implements __torch_dispatch__
    is 'traceable' with torch.compile.
    In order for a tensor subclass to support TorchDispatchMode-style tracing in PT2,
    It must implement two magic methods: __tensor_flatten__ and __tensor_unflatten__.
    It is also expected to obey some restrictions around traceability and aliasing
    (TODO: add clear documentation around this.)
    """
    is_subclass = isinstance(t, torch.Tensor) and type(t) != torch.Tensor
    return is_subclass and hasattr(t, '__tensor_flatten__') and hasattr(t, '__tensor_unflatten__')
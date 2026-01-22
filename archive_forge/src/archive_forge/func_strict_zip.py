import dataclasses
import warnings
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import py_sym_types
def strict_zip(*iterables, strict=True, **kwargs):
    if not strict:
        return original_zip(*iterables, **kwargs)
    shortest_length = min((len(it) for it in iterables))
    for iterable in iterables:
        if len(iterable) != shortest_length:
            raise ValueError('The iterables have different lengths and strict mode is enabled.')
    return original_zip(*iterables, **kwargs)
import dataclasses
import warnings
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import py_sym_types
def normalize_as_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]
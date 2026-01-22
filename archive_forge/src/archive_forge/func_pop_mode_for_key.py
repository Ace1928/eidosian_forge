import contextlib
import ctypes
import importlib
import inspect
import sys
import types
from typing import Any, Callable, Dict, List, Type, Union
import torch._C
import torch.utils._pytree as pytree
from torch import _utils_internal
from torch._functorch.pyfunctorch import dispatch_functorch
def pop_mode_for_key(key):
    assert isinstance(key, torch._C.DispatchKey)
    assert key in mode_stack_per_key()
    curr_mode_stack = mode_stack_per_key()[key]
    assert len(curr_mode_stack) > 0
    return curr_mode_stack.pop()
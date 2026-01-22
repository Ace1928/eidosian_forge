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
def py_impl(self, k):
    if isinstance(k, torch._C.DispatchKey) and (not self.non_fallthrough_keys.has(k)):
        self.non_fallthrough_keys = self.non_fallthrough_keys.add(k)
    return super().py_impl(k)
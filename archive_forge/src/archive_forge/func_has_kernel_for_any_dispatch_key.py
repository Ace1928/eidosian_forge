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
def has_kernel_for_any_dispatch_key(self, ks):
    return torch._C._dispatch_has_kernel_for_any_dispatch_key(self.name(), ks) or super().has_kernel_for_any_dispatch_key(ks)
import contextlib
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch.fx import Tracer, GraphModule
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily, is_fake
from torch._dispatch.python import enable_python_dispatcher, enable_pre_dispatch
import torch.fx as fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from contextlib import contextmanager, nullcontext
import inspect
from dataclasses import dataclass
import weakref
import operator
from torch.utils._stats import count
import logging
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import (
from .sym_node import SymNode
from ._sym_dispatch_mode import SymDispatchMode
from torch.fx import Proxy
import torch.fx.traceback as fx_traceback
from torch import SymInt, SymFloat, SymBool
from torch.utils.weak import WeakTensorKeyDictionary
def create_arg(self, a: Any):
    if isinstance(a, torch.nn.Parameter):
        for n, p in self.root.named_parameters():
            if a is p:
                return self.create_node('get_attr', n, (), {})
        qualname: Optional[str] = None
        if not qualname:
            i = 0
            while True:
                qualname = f'_param_constant{i}'
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)
        return self.create_node('get_attr', qualname, (), {})
    elif isinstance(a, (SymInt, SymFloat, SymBool)):
        assert a.node.constant is not None
        return a.node.constant
    return super().create_arg(a)
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
def track_tensor_tree(inner_res, proxy_res, *, constant, tracer):

    def wrap_with_proxy(e, proxy, constant):
        if isinstance(e, torch.Tensor):
            track_tensor(e, proxy, tracer=tracer, constant=constant)
            set_meta(proxy, e)
        elif isinstance(e, py_sym_types):
            set_meta(proxy, e)
            set_proxy_slot(e.node, tracer, lambda: proxy)
        elif isinstance(e, (tuple, list)):
            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)
            for idx, ee in enumerate(e):
                wrap_with_proxy(ee, proxy[idx], get_constant(idx))
        elif isinstance(e, dict):
            assert constant is None
            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)
            for key, val in e.items():
                wrap_with_proxy(val, proxy[key], None)
        else:
            pass

    def get_constant(idx):
        if constant is None:
            return None
        else:
            return constant[idx]
    wrap_with_proxy(inner_res, proxy_res, constant)
    return inner_res
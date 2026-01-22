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
def track_tensor(tensor, proxy, *, constant, tracer):

    def try_set_proxy_slot(outer_s, proxy_callable, *args):
        assert callable(proxy_callable)
        if isinstance(outer_s, SymInt):
            inner_s = outer_s.node
            set_proxy_slot(inner_s, tracer, thunkify(proxy_callable, outer_s, *args))
    for i, s in enumerate(tensor.shape):
        try_set_proxy_slot(s, lambda x, i: set_meta(torch.ops.aten.sym_size.int(proxy, i), x), i)
    for i, s in enumerate(tensor.stride()):
        try_set_proxy_slot(s, lambda x, i: set_meta(torch.ops.aten.sym_stride.int(proxy, i), x), i)
    try_set_proxy_slot(tensor.numel(), lambda x: set_meta(torch.ops.aten.sym_numel.default(proxy), x))
    try_set_proxy_slot(tensor.storage_offset(), lambda x: set_meta(torch.ops.aten.sym_storage_offset.default(proxy), x))
    set_proxy_slot(tensor, tracer, _ProxyTensor(proxy, constant))
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
def make_fx(f, decomposition_table=None, tracing_mode='real', _allow_non_fake_inputs=False, *, pre_dispatch=False, _allow_fake_constant=False, _error_on_data_dependent_ops=True):
    assert tracing_mode in ['real', 'fake', 'symbolic']
    if decomposition_table is None:
        decomposition_table = {}

    @functools.wraps(f)
    def wrapped(*args):
        from .symbolic_shapes import ShapeEnv
        phs = pytree.tree_map(lambda _: fx.PH, args)
        fx_tracer = PythonKeyTracer()
        fake_tensor_mode: Any = nullcontext()
        if tracing_mode == 'real':
            fake_tensor_mode = nullcontext()
        elif tracing_mode == 'fake':
            import torch._dynamo
            fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
            if fake_tensor_mode is None:
                fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True, allow_non_fake_inputs=_allow_non_fake_inputs, shape_env=ShapeEnv(), static_shapes=True)
        elif tracing_mode == 'symbolic':
            import torch._dynamo
            fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
            if fake_tensor_mode is None:
                shape_env = ShapeEnv()
                fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=_allow_non_fake_inputs, shape_env=shape_env)
            else:
                shape_env = fake_tensor_mode.shape_env
                assert shape_env is not None, "shape_env should be set if tracing with 'symbolic'"
        else:
            raise AssertionError(f'Unexpected tracing type: {tracing_mode}')
        python_dispatcher_mode: Any = nullcontext()
        pre_dispatch_mode: Any = nullcontext()
        if tracing_mode == 'symbolic' or pre_dispatch:
            python_dispatcher_mode = enable_python_dispatcher()
        if pre_dispatch:
            pre_dispatch_mode = enable_pre_dispatch()
        proxy_function_mode: Any = nullcontext()
        if pre_dispatch:
            proxy_function_mode = PreDispatchTorchFunctionMode(fx_tracer)
        proxy_mode = ProxyTorchDispatchMode(fx_tracer, tracing_mode, pre_dispatch=pre_dispatch, _allow_fake_constant=_allow_fake_constant, _error_on_data_dependent_ops=_error_on_data_dependent_ops)
        arg_count = 0

        def wrap_fake(x):
            nonlocal arg_count
            from torch._dynamo.source import ConstantSource
            source = ConstantSource(f'input{arg_count}')
            if isinstance(x, torch.Tensor):
                arg_count += 1
                return fake_tensor_mode.from_tensor(x, source=source)
            elif type(x) is int and tracing_mode == 'symbolic':
                return shape_env.create_symintnode(shape_env.create_symbol(x, source, positive=None), hint=x, source=source)
            return x
        sym_mode = proxy_mode.sym_mode
        wrap_fn_map = {'real': lambda x: x, 'fake': wrap_fake, 'symbolic': wrap_fake}
        args = pytree.tree_map(wrap_fn_map[tracing_mode], args)
        if not hasattr(inspect.unwrap(f), '__code__') or inspect.unwrap(f).__code__.co_flags & inspect.CO_VARARGS:
            func = fake_signature(f, len(phs))
        else:
            func = f
        with decompose(decomposition_table), fake_tensor_mode, python_dispatcher_mode, pre_dispatch_mode, proxy_function_mode, sym_mode, proxy_mode, disable_autocast_cache():
            t = dispatch_trace(wrap_key(func, args, fx_tracer, pre_dispatch), tracer=fx_tracer, concrete_args=tuple(phs))
        if tracing_mode == 'symbolic':
            t.shape_env = shape_env
        return t
    return wrapped
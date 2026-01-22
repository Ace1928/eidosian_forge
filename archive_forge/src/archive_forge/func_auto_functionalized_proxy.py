from typing import Any, Dict, List, Tuple
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
@auto_functionalized.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_proxy(mode, op: torch._ops.OpOverload, mutated_args_names: List[str], kwargs: Dict[str, Any]) -> Tuple[Tensor, ...]:
    if not mode.enable_tracing:
        return auto_functionalized(op, mutated_args_names, kwargs)
    with disable_proxy_modes_tracing():
        out = auto_functionalized(op, mutated_args_names, kwargs)
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
    out_proxy = mode.tracer.create_proxy('call_function', auto_functionalized, (op, mutated_args_names, proxy_kwargs), {})
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result
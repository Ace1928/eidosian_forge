import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import (
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._higher_order_ops.utils import autograd_not_implemented
@out_dtype.py_impl(ProxyTorchDispatchMode)
def out_dtype_proxy(mode: ProxyTorchDispatchMode, op: torch._ops.OpOverload, output_dtype: torch.dtype, *args):
    if mode.enable_tracing:
        return trace_out_dtype(mode, out_dtype, op, output_dtype, *args)
    else:
        return out_dtype(op, output_dtype, *args)
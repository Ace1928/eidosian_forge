import time
import io
from typing import Dict, List, Tuple, Any
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed.rpc import RRef
from torch.distributed.rpc.internal import RPCExecMode, _build_rpc_profiling_key
from torch.futures import Future
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.autograd.profiler_legacy import profile as _profile
@torch.jit.script
def script_rpc_async_call_with_assorted_types(dst_worker_name: str):
    args = (torch.tensor([1, 1]), 'str_arg', 1)
    kwargs: Dict[str, Any] = {'tensor_kwarg': torch.tensor([3, 3]), 'str_kwarg': '_str_kwarg', 'int_kwarg': 3}
    fut = rpc.rpc_async(dst_worker_name, assorted_types_args_kwargs, args, kwargs)
    ret = fut.wait()
    return ret
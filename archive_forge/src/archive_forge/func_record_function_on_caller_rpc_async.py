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
def record_function_on_caller_rpc_async(dst_worker_name: str, block: str) -> Tensor:
    t: Tensor = torch.ones(1)
    with record_function(block) as rf:
        fut1 = rpc.rpc_async(dst_worker_name, script_add_ones, (t,))
        zero = torch.zeros_like(t)
        fut2 = rpc.rpc_async(dst_worker_name, script_add_ones, (t,))
        res = fut1.wait() + fut2.wait() + zero
    return res
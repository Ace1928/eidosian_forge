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
@dist_init
def test_more_than_needed_args_are_specified(self):
    if self.rank != 0:
        return
    dst_worker_name = worker_name((self.rank + 1) % self.world_size)
    with self.assertRaisesRegex(RuntimeError, 'Expected at most 4 arguments but found 5 positional arguments'):

        @torch.jit.script
        def script_rpc_async_call_with_more_args(dst_worker_name: str):
            args = (torch.tensor([1, 1]), torch.tensor([2, 2]), torch.tensor([3, 3]), torch.tensor([4, 4]), torch.tensor([5, 5]))
            kwargs = {}
            fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
            ret = fut.wait()
            return ret
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
def test_callback_chain(self):
    n = self.rank + 1
    dst = worker_name(n % self.world_size)

    def callback(fut):
        return fut.wait() + 1
    fut = rpc.rpc_async(worker_name(n % self.world_size), one_arg, args=(torch.ones(n, n),))
    num_cbs = 20
    for _ in range(num_cbs):
        fut = fut.then(callback)
    self.assertEqual(fut.wait(), torch.ones(n, n) + 1 + num_cbs)
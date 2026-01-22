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
def test_callback_with_exception(self):

    def callback(fut):
        with self.assertRaisesRegex(Exception, '.*Expected error.*'):
            fut.wait()
        raise RuntimeError('Another expected error')
    future = rpc.rpc_async(worker_name((self.rank + 1) % self.world_size), script_fork_wait_throw, args=(torch.ones(2),)).then(callback)
    with self.assertRaisesRegex(RuntimeError, 'Another expected error'):
        future.wait()
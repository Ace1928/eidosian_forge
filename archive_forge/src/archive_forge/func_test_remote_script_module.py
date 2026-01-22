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
def test_remote_script_module(self):
    import torch.distributed.rpc.api as api
    api._ignore_rref_leak = True
    local_ret = torch.ones(self.rank) + torch.ones(self.rank)
    n = self.rank + 1
    dst_rank = n % self.world_size
    remote_ref = rpc.remote(worker_name(dst_rank), construct_my_script_module, args=(self.rank,))
    ret = rpc.rpc_sync(worker_name(dst_rank), run_ref_script_module, args=(remote_ref, torch.ones(self.rank)))
    self.assertEqual(ret, local_ret)
    with self.assertRaisesRegex(RuntimeError, "is an RRef to a ScriptModule. It can't be sent through RPC from owner,"):
        ret = rpc.rpc_sync(worker_name(self.rank), run_ref_script_module, args=(remote_ref, torch.ones(self.rank)))
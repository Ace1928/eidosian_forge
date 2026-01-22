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
def test_rpc_async_jit_profiled(self):
    if self.rank == 0:
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker_name = worker_name(dst_rank)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {}
        with _profile() as prof:
            script_rpc_async_call(dst_worker_name, args, kwargs)
        function_events = prof.function_events
        qual_name = torch._jit_internal._qualified_name(two_args_two_kwargs)
        rpc_async_jit_event = [event for event in function_events if qual_name in event.name and event.node_id == self.rank]
        self.assertEqual(len(rpc_async_jit_event), 1)
        rpc_async_jit_event = rpc_async_jit_event[0]
        profiled_name = _build_rpc_profiling_key(RPCExecMode.ASYNC_JIT, qual_name, worker_name(self.rank), dst_worker_name)
        self.assertEqual(profiled_name, rpc_async_jit_event.name)
        remote_events = [event for event in function_events if event.is_remote]
        remote_event_node_ids = {remote_event.node_id for remote_event in remote_events}
        self.assertEqual(remote_event_node_ids, {dst_rank})
        remote_add = next((remote_event for remote_event in remote_events if 'aten::add' in remote_event.name))
        remote_add_profiled_name = f'{profiled_name}#remote_op: aten::add'
        self.assertEqual(remote_add.name, remote_add_profiled_name)
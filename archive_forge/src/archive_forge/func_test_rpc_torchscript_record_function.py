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
def test_rpc_torchscript_record_function(self):
    REMOTE_OP_STR = '#remote_op: '
    if self.rank == 0:
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker_name = worker_name(dst_rank)
        block_scope = 'foo'
        with _profile() as prof:
            call_rpc_torchscript_with_record_function(dst_worker_name, block_scope)
        prof.key_averages()
        function_events = prof.function_events
        expected_key = _build_rpc_profiling_key(RPCExecMode.ASYNC_JIT, torch._jit_internal._qualified_name(script_add_ones_with_record_function), worker_name(self.rank), dst_worker_name) + REMOTE_OP_STR + block_scope
        remote_record_function_event = next((evt for evt in function_events if evt.name == expected_key))
        self.assertTrue(block_scope in remote_record_function_event.name)
        remote_children = remote_record_function_event.cpu_children
        self.assertTrue(('aten::add' in child.name for child in remote_children))
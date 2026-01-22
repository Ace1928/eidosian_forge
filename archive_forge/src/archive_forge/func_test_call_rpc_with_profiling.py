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
def test_call_rpc_with_profiling(self):
    if self.rank == 0:
        with _profile() as prof:
            prof_key = _build_rpc_profiling_key(RPCExecMode.ASYNC, torch._jit_internal._qualified_name(one_arg), 'worker0', 'worker1')
            with torch.autograd.profiler.record_function(prof_key) as rf:
                ret = call_rpc_with_profiling(rf.record, 'worker1')
        events = prof.function_events
        function_event = get_function_event(events, prof_key)
        self.assertTrue(torch._jit_internal._qualified_name(one_arg) in function_event.name)
import concurrent.futures
import contextlib
import json
import os
import sys
import threading
import time
from collections import namedtuple
from functools import partial
from threading import Event
from threading import Lock
from unittest import mock
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info, WorkerInfo
from torch.distributed.rpc.api import _use_rpc_pickler, _thread_local_var, _wait_all
from torch.distributed.rpc.internal import (
from torch.futures import Future
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_utils import TemporaryFileName
from torch.autograd.profiler_legacy import profile as _profile
def rpc_with_profiling(dst_worker):
    with _profile() as prof:
        fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
        fut.wait()
    events = prof.function_events
    remote_event_names = {event.name: event for event in events if event.is_remote}
    rpc_profiling_key = _build_rpc_profiling_key(RPCExecMode.ASYNC, udf_with_torch_ops.__qualname__, worker_name(self.rank), dst_worker)
    remote_event_name_set = set(EXPECTED_REMOTE_EVENTS)
    for name, event in remote_event_names.items():
        self.assertTrue(name.startswith(rpc_profiling_key))
        self.assertTrue(event.is_remote)
        self.assertTrue(event.node_id == rpc.get_worker_info(dst_worker).id)
        operator_name_substr = name[len(rpc_profiling_key):]
        matching_event = {remote_event_name for remote_event_name in remote_event_name_set if remote_event_name in operator_name_substr}
        remote_event_name_set -= matching_event
    self.assertTrue(remote_event_name_set == set(), f'Expected {remote_event_name_set} to be included in remote profiler output.')
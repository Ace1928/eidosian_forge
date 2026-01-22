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
@dist_init
def test_rpc_profiling_remote_record_function(self):
    if self.rank != 1:
        return
    dst_ranks = [i for i in range(self.world_size) if i != self.rank]
    for dst_rank in dst_ranks:
        dst_worker = worker_name(dst_rank)
        with _profile() as prof:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=(-1, True))
            fut.wait()
        function_events = prof.function_events
        record_function_remote_event = [evt for evt in function_events if '##forward##' in evt.name]
        self.assertEqual(1, len(record_function_remote_event))
        record_function_remote_event = record_function_remote_event[0]
        self.assertEqual(record_function_remote_event.node_id, dst_rank)

        def get_cpu_children(event):
            if not event.cpu_children:
                return []
            cpu_children = event.cpu_children
            for e in event.cpu_children:
                cpu_children.extend(get_cpu_children(e))
            return cpu_children
        remote_children = get_cpu_children(record_function_remote_event)
        with _profile() as prof:
            udf_with_torch_ops(-1, True)
        local_function_events = prof.function_events
        local_record_function_event = next((evt for evt in local_function_events if '##forward##' in evt.name))
        local_children = get_cpu_children(local_record_function_event)
        local_children_names = [evt.name for evt in local_children]
        REMOTE_OP_STR = '#remote_op: '

        def convert_remote_to_local(event_name):
            remote_op_key = REMOTE_OP_STR
            return event_name[event_name.find(remote_op_key) + len(remote_op_key):]
        for evt in remote_children:
            local_name = convert_remote_to_local(evt.name)
            self.assertTrue(local_name in local_children_names)
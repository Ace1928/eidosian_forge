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
@dist_init(setup_rpc=False)
def test_wait_all_workers_timeout(self):
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
    og_func = rpc.api._wait_all_workers

    def wait_all_workers_sleep(timeout):
        rpc.api._all_gather(SlowPickleClass(0.5), timeout=timeout)
    rpc.api._wait_all_workers = wait_all_workers_sleep
    try:
        with self.assertRaisesRegex(RuntimeError, ''):
            rpc.shutdown(graceful=True, timeout=0.01)
    finally:
        rpc.api._wait_all_workers = og_func
    dist.barrier()
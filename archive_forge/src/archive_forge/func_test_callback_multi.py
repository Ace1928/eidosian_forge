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
def test_callback_multi(self):
    num_cbs = 10
    n = self.rank + 1

    def callback(idx, fut):
        ret = fut.wait()
        self.assertEqual(ret, torch.ones(n, n) * 2)
        return ret + idx
    fut = rpc.rpc_async(worker_name(n % self.world_size), torch.add, args=(torch.ones(n, n), torch.ones(n, n)))
    cb_futs = []
    for idx in range(num_cbs):
        cb_futs.append(fut.then(partial(callback, idx)))
    self.assertEqual(fut.wait(), torch.ones(n, n) * 2)
    for idx in range(num_cbs):
        self.assertEqual(cb_futs[idx].wait(), torch.ones(n, n) * 2 + idx)
    self.assertEqual(fut.wait(), torch.ones(n, n) * 2)
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
def test_wait_all(self):
    with _wait_all():
        self.assertTrue(_thread_local_var.future_list == [])
        dst = worker_name((self.rank + 1) % self.world_size)
        fut = rpc.rpc_async(dst, torch.add, (torch.ones(2, 2), 1))
        self.assertTrue(len(_thread_local_var.future_list) == 1)
        self.assertTrue(isinstance(_thread_local_var.future_list[0], torch._C.Future))
    self.assertTrue(fut.done())
    self.assertEqual(fut.wait(), torch.ones(2, 2) + 1)
    self.assertFalse(hasattr(_thread_local_var, 'future_list'))
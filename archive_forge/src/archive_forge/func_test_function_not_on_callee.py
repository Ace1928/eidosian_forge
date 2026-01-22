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
def test_function_not_on_callee(self):
    this_module = sys.modules[__name__]
    caller_worker = 'worker0'
    callee_worker = 'worker1'
    if self.rank == 1:
        delattr(this_module, 'foo_add')
        rpc.rpc_sync(caller_worker, set_value, args=(self.rank,))
    if self.rank == 0:
        wait_for_value_future()
        self.assertTrue(hasattr(this_module, 'foo_add'))
        with self.assertRaisesRegex(RuntimeError, 'RPC pickler does not serialize'):
            rpc.rpc_sync(callee_worker, foo_add, args=())
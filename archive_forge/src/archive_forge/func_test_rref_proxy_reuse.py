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
def test_rref_proxy_reuse(self):
    rref = rpc.remote(worker_name((self.rank + 1) % self.world_size), my_function, args=(torch.ones(2, 2), 1, 3))
    expected = torch.ones(2, 2) + 1 + 3
    proxy_rpc_sync = rref.rpc_sync()
    proxy_rpc_async = rref.rpc_async()
    proxy_remote = rref.remote()
    self.assertEqual(expected.size(), proxy_rpc_sync.size())
    self.assertEqual(expected + 1, proxy_rpc_sync.add(1))
    self.assertEqual(expected.view(1, 4), proxy_rpc_sync.view(1, 4))
    self.assertEqual(expected.size(), proxy_rpc_async.size().wait())
    self.assertEqual(expected + 3, proxy_rpc_async.add(3).wait())
    self.assertEqual(expected.view(4, 1), proxy_rpc_async.view(4, 1).wait())
    self.assertEqual(expected.size(), proxy_remote.size().to_here())
    self.assertEqual(expected + 5, proxy_remote.add(5).to_here())
    self.assertEqual(expected.view(-1), proxy_remote.view(-1).to_here())
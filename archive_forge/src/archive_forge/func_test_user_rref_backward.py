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
def test_user_rref_backward(self):
    dst = worker_name((self.rank + 1) % self.world_size)
    t = torch.rand(10, requires_grad=True)
    with dist_autograd.context() as context_id:
        rref = rpc.remote(dst, RpcTest._sum, args=(t,))
        rref.backward(context_id, retain_graph=True)
        rref.backward(context_id)
        self.assertEqual(torch.ones_like(t) * 2, dist_autograd.get_gradients(context_id)[t])
    with dist_autograd.context() as context_id:
        rref = rpc.remote(dst, RpcTest._identity, args=('foo',))
        with self.assertRaisesRegex(RuntimeError, 'RRef should contain a tensor for .backward()'):
            rref.backward(context_id)
        with self.assertRaisesRegex(RuntimeError, "User RRefs require 'dist_autograd_ctx_id' to be specified"):
            rref.backward()
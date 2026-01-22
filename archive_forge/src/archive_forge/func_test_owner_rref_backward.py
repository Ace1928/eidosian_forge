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
def test_owner_rref_backward(self):
    dst = worker_name((self.rank + 1) % self.world_size)
    t1 = torch.rand(10, 10, requires_grad=True)
    rref = rpc.RRef(t1.sum() + t1.sum())
    rref.backward()
    expected_grad = torch.ones_like(t1) * 2
    self.assertEqual(expected_grad, t1.grad)
    with dist_autograd.context() as context_id:
        t2 = rpc.rpc_sync(dst, torch.add, args=(t1, t1))
        rref = rpc.RRef(t2.sum())
        rref.backward(context_id)
        self.assertEqual(expected_grad, dist_autograd.get_gradients(context_id)[t1])
    with dist_autograd.context() as context_id:
        t2 = rpc.rpc_sync(dst, torch.add, args=(t1, t1))
        rref = rpc.RRef(t2.sum())
        rref.backward(context_id, retain_graph=True)
        rref.backward(context_id)
        self.assertEqual(expected_grad * 2, dist_autograd.get_gradients(context_id)[t1])
    with self.assertRaisesRegex(RuntimeError, 'tensors does not require grad and does not have a grad_fn'):
        rpc.RRef(torch.rand(10)).backward()
    with self.assertRaisesRegex(RuntimeError, 'grad can be implicitly created only for scalar outputs'):
        rpc.RRef(torch.rand(10, requires_grad=True)).backward()
    with self.assertRaisesRegex(RuntimeError, 'Could not find autograd context with id: 100'):
        rpc.RRef(torch.rand(10, requires_grad=True).sum()).backward(100)
    with self.assertRaisesRegex(RuntimeError, 'RRef should contain a tensor for .backward()'):
        rpc.RRef('foo').backward()
import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
@skip_if_lt_x_gpu(4)
def test_device_maps_backward_pass(self):
    options = self.rpc_backend_options
    dst = worker_name((self.rank + 1) % self.world_size)
    options.set_device_map(dst, {self.rank: (self.rank + 1) % self.world_size})
    rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
    t1 = torch.rand(10, device=self.rank, requires_grad=True)
    t2 = torch.rand(10, device=self.rank, requires_grad=True)
    with dist_autograd.context() as context_id:
        res = rpc.rpc_sync(dst, torch.add, args=(t1, t2))
        dist_autograd.backward(context_id, [res.sum()])
        grads = dist_autograd.get_gradients(context_id)
        self.assertEqual(torch.ones(10), grads[t1])
        self.assertEqual(torch.ones(10), grads[t2])
        self.assertEqual(t1.device, grads[t1].device)
        self.assertEqual(t2.device, grads[t2].device)
    rpc.shutdown()
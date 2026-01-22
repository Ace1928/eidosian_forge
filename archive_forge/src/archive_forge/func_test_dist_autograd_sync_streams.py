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
def test_dist_autograd_sync_streams(self):
    options = self.rpc_backend_options
    dst = worker_name((self.rank + 1) % self.world_size)
    options.set_device_map(dst, {self.rank: (self.rank + 1) % self.world_size})
    rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
    remote_compute = rpc.remote(dst, TensorPipeCudaDistAutogradTest.MyRemoteCompute)
    local_compute = TensorPipeCudaDistAutogradTest.MyLocalCompute(remote_compute)
    for _ in range(10):
        input = torch.rand([1000, 10000], device=self.rank, requires_grad=True)
        result = input * 2.0
        r = random.random()
        loss = result.sum() * r
        loss.backward()
        with dist_autograd.context() as context_id:
            result = local_compute(input)
            loss = result.sum() * r
            dist_autograd.backward(context_id, [loss])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(input.grad, grads[input])
    rpc.shutdown()
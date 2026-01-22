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
@dist_init
def test_thread_local_context_id(self):
    t1 = torch.rand((3, 3))
    t2 = torch.rand((3, 3))
    t3 = t1 + t2
    t3.requires_grad = True
    t3.sum().backward()
    dst = worker_name((self.rank + 1) % self.world_size)
    rref = rpc.remote(dst, DistAutogradTest._slow_add, args=(t1, t2))
    with dist_autograd.context() as context_id:
        loss = rref.to_here().sum()
        dist_autograd.backward(context_id, [loss])
        self.assertTrue(rpc.rpc_sync(dst, _compare_owner_value, args=(context_id, rref, t3.grad)))
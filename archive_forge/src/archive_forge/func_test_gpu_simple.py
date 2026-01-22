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
@skip_if_lt_x_gpu(1)
@dist_init
def test_gpu_simple(self):
    t1 = torch.rand(3, 3, requires_grad=True, device='cuda:0')
    t2 = torch.rand(3, 3, requires_grad=True, device='cuda:0')
    (t1 + t2).sum().backward()
    with dist_autograd.context() as context_id:
        t3 = t1 + t2
        dist_autograd.backward(context_id, [t3.sum()])
        grads = dist_autograd.get_gradients(context_id)
        self.assertEqual(2, len(grads))
        self.assertEqual(t1.grad, grads[t1])
        self.assertEqual(t2.grad, grads[t2])
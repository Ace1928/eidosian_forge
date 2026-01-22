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
def test_backward_accumulate_grads(self):
    t1 = torch.rand((3, 3), requires_grad=True)
    t2 = torch.rand((3, 3), requires_grad=True)
    with dist_autograd.context() as context_id:
        t3 = torch.matmul(t1, t2)
        torch.autograd.backward([t3.sum()], retain_graph=True)
        torch.autograd.backward([t3.sum()])
        t3 = rpc.rpc_sync(worker_name(self._next_rank()), torch.matmul, args=(t1, t2))
        dist_autograd.backward(context_id, [t3.sum()], retain_graph=True)
        dist_autograd.backward(context_id, [t3.sum()])
        grads = dist_autograd.get_gradients(context_id)
        self.assertEqual(2, len(grads))
        self.assertIn(t1, grads)
        self.assertIn(t2, grads)
        self.assertEqual(t1.grad, grads[t1])
        self.assertEqual(t2.grad, grads[t2])
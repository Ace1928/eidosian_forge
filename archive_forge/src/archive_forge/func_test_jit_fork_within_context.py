from typing import Dict, Tuple
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import rpc_async
from torch.testing import FileCheck
from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init
def test_jit_fork_within_context(self):
    with dist_autograd.context() as context_id:
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        res = fork_add(t1, t2, dst_worker_name)
        loss = res.sum()
        dist_autograd.backward(context_id, [loss])
        grads = dist_autograd.get_gradients(context_id)
        self.assertEqual(2, len(grads))
        self.assertIn(t1, grads)
        self.assertIn(t2, grads)
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
def test_dist_backward(self):
    if self.rank != 0:
        return

    @torch.jit.script
    def dist_backward_script(context_id: int, loss: torch.Tensor):
        dist_autograd.backward(context_id, [loss])
    FileCheck().check('dist_backward').run(str(dist_backward_script.graph))
    with dist_autograd.context() as context_id:
        t1 = torch.rand(3, 3, requires_grad=True)
        t2 = torch.rand(3, 3, requires_grad=True)
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        loss = rpc.rpc_sync(dst_worker_name, torch.add, args=(t1, t2)).sum()
        dist_backward_script(context_id, loss)
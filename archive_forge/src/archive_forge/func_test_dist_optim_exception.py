import threading
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init()
def test_dist_optim_exception(self):
    owner1 = 'worker%d' % ((self.rank + 1) % self.world_size)
    owner2 = 'worker%d' % ((self.rank + 2) % self.world_size)
    remote_module1 = rpc.remote(owner1, MyModule)
    remote_module2 = rpc.remote(owner2, MyModule)
    remote_param1 = remote_method(MyModule.get_w, remote_module1)
    remote_param2 = remote_method(MyModule.get_w, remote_module2)
    dist_optim = DistributedOptimizer(FailingOptimizer, [remote_param1, remote_param2])
    with dist_autograd.context() as context_id:
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        output1 = rpc_async_method(MyModule.forward, remote_module1, t2)
        output2 = rpc_async_method(MyModule.forward, remote_module2, output1.wait())
        loss = torch.add(output2.wait(), t1).sum()
        dist_autograd.backward(context_id, [loss])
        with self.assertRaisesRegex(Exception, 'Error running optimizer'):
            dist_optim.step(context_id)
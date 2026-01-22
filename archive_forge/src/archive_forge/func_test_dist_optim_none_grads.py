import threading
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init()
def test_dist_optim_none_grads(self):
    self._test_dist_optim_none_grads(optim.SGD, lr=0.05)
    self._test_dist_optim_none_grads(optim.RMSprop, lr=0.05)
    self._test_dist_optim_none_grads(optim.Rprop, lr=0.05)
    self._test_dist_optim_none_grads(optim.Adadelta, rho=0.95)
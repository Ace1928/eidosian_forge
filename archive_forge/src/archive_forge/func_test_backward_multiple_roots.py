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
def test_backward_multiple_roots(self):
    local_grads = None
    t1 = torch.rand((3, 3), requires_grad=True)
    t2 = torch.rand((3, 3), requires_grad=True)
    for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC]:
        with dist_autograd.context() as context_id:
            r1 = self._exec_func(exec_mode, torch.add, t1, t2).sum()
            r2 = self._exec_func(exec_mode, torch.mul, t1, t2).sum()
            r3 = self._exec_func(exec_mode, torch.cos, t1).sum()
            r4 = self._exec_func(exec_mode, torch.div, t1, t2).sum()
            local_grads = self._verify_backwards(exec_mode, [r1, r2, r3, r4], context_id, local_grads, t1, t2)
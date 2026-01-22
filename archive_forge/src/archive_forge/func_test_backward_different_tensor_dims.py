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
def test_backward_different_tensor_dims(self):
    local_grads = None
    t1 = torch.rand((4, 6), requires_grad=True)
    t2 = torch.rand((6, 5))
    t3 = torch.rand((5, 7), requires_grad=True)
    t4 = torch.rand((7, 9))
    for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
        with dist_autograd.context() as context_id:
            val = self._exec_func(exec_mode, torch.matmul, t1, t2)
            val = self._exec_func(exec_mode, torch.linalg.multi_dot, (val, t3, t4))
            loss = val.sum()
            ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2, t2, t3, t4)
            local_grads = ret if ret else local_grads
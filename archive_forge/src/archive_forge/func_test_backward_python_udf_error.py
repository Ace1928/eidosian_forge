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
def test_backward_python_udf_error(self):
    t1 = torch.rand((3, 3), requires_grad=True)
    t2 = torch.rand((3, 3), requires_grad=True)
    with dist_autograd.context() as context_id:
        loss = rpc.rpc_sync(worker_name(self._next_rank()), DistAutogradTest._nested_rpc_call_backward_error, args=(t1, t2, self._next_rank()))
        with self.assertRaisesRegex(RuntimeError, 'Simulate error on backward pass'):
            dist_autograd.backward(context_id, [loss.sum()])
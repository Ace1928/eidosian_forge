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
def test_backward_rref_nested_sparse(self):
    callee = worker_name((self.rank + 1) % self.world_size)
    rref_owner = worker_name((self.rank + 2) % self.world_size)
    self._backward_rref(callee, rref_owner, build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), None, True)
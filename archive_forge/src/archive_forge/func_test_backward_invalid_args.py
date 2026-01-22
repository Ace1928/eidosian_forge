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
def test_backward_invalid_args(self):
    with dist_autograd.context() as context_id:
        with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
            dist_autograd.backward(context_id, None)
        with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
            dist_autograd.backward(None, None)
        with self.assertRaisesRegex(RuntimeError, 'No tensors provided for gradient computation'):
            dist_autograd.backward(context_id, [])
        with self.assertRaisesRegex(RuntimeError, 'requires_grad not set on'):
            t = torch.rand(3, 3)
            dist_autograd.backward(context_id, [t])
        with self.assertRaisesRegex(RuntimeError, 'is not a scalar, all roots need to be scalar'):
            t = torch.rand(3, 3, requires_grad=True)
            dist_autograd.backward(context_id, [t])
        with self.assertRaisesRegex(RuntimeError, 'does not have a valid gradient function'):
            t = torch.rand(1, requires_grad=True)
            dist_autograd.backward(context_id, [t])
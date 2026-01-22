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
def test_grad_copy_sparse_indices_extra_ref(self):

    class MyFunc(Function):
        static_grad_ptr = None
        static_grad_indices_ref = None
        static_grad_values_ref = None

        @staticmethod
        def forward(ctx, inp):
            return inp

        @staticmethod
        def backward(ctx, grad):
            MyFunc.static_grad_ptr = grad._values().data_ptr()
            MyFunc.static_grad_indices_ref = grad._indices()
            MyFunc.static_grad_values_ref = grad._values()
            return grad
    a = torch.randn(10, 3, requires_grad=True)
    input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
    offsets = torch.tensor([0, 4])
    import torch.nn.functional as F
    with dist_autograd.context() as context_id:
        emb_matrix = MyFunc.apply(a)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        dist_autograd.backward(context_id, [loss], retain_graph=True)
        grads = dist_autograd.get_gradients(context_id)
        p_g = MyFunc.static_grad_ptr
        p_a = grads[a]._values().data_ptr()
        self.assertIsNotNone(MyFunc.static_grad_indices_ref)
        self.assertIsNotNone(MyFunc.static_grad_values_ref)
        self.assertTrue(p_g == p_a)
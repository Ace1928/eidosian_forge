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
def test_clean_context_during_backward(self):
    """
        This test simulates the situation where the 'backward' call might throw
        an exception locally which would lead to the autograd context being
        cleaned up if we're using the context manager. As a result, the autograd
        context might be cleaned up while some threads are still using the
        autograd context.

        It is fine for the 'backward' call to throw an exception in this test,
        but the process should not crash.
        """
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    context = dist_autograd._new_context()
    context_id = context._context_id()
    DistAutogradTest._test_clean_context_backward_context_id = context_id
    for i in range(0, self.world_size):
        if i != self.rank:
            rank_distance = (i - self.rank + self.world_size) % self.world_size
            rpc.rpc_sync(worker_name(i), _set_rpc_done, args=(context_id, rank_distance))
    dist.barrier()
    self.assertEqual(self.world_size - 1, len(known_context_ids))
    t1 = torch.rand((3, 3), requires_grad=True)
    for i in range(0, 100):
        dst = self._next_rank()
        t1 = rpc.rpc_sync(worker_name(dst), torch.add, args=(t1, t1))
    t1 = DistAutogradTest.MyBackwardFunc.apply(t1)
    self.assertEqual(100, len(context._send_functions()))
    context_id = 100
    with self.assertRaisesRegex(RuntimeError, f'Could not find autograd context with id: {context_id}'):
        dist_autograd.backward(context_id, [t1.sum()])
    dist.barrier()
    rpc.shutdown(graceful=False)
    sys.exit(0)
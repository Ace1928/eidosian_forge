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
def context_cleanup_test_helper(self, rpc_args, func):
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}
    with dist_autograd.context() as context_id:
        for dst_rank in dst_ranks:
            rpc.rpc_sync(worker_name(dst_rank), func, args=rpc_args)
            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
    with self.assertRaises(RuntimeError):
        dist_autograd._retrieve_context(context_id)
    dist.barrier()
    success = _all_contexts_cleaned_up()
    self.assertTrue(success)
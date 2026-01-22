from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
def compute_q(fut):
    state.p_memory_dict[bucket_index] = fut.value()[0]
    _orthogonalize(state.p_memory_dict[bucket_index])
    torch.matmul(matrix.t(), state.p_memory_dict[bucket_index], out=state.q_memory_dict[bucket_index])
    return dist.all_reduce(state.q_memory_dict[bucket_index], group=group_to_use, async_op=True).get_future().wait()[0]
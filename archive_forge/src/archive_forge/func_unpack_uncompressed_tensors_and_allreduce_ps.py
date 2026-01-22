from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
def unpack_uncompressed_tensors_and_allreduce_ps(fut):
    uncompressed_tensors_memory = fut.value()[0].div_(world_size)
    idx = 0
    for tensor in uncompressed_tensors:
        tensor.copy_(uncompressed_tensors_memory[idx:idx + tensor.numel()].view_as(tensor))
        idx += tensor.numel()
    return dist.all_reduce(state.p_memory_dict[bucket_index], group=group_to_use, async_op=True).get_future().wait()[0]
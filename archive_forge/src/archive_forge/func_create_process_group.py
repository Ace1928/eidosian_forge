import collections
import logging
import sys
from typing import Any, Dict, List, MutableMapping, Set, Tuple
import torch
import torch.distributed as dist
def create_process_group(ranks: List[int]) -> torch.distributed.ProcessGroup:
    """
    Creates and intializes a new process group. Assumes init_process_group
    has already been called
    Arguments:
        ranks (list<int>): ranks corresponding to the processes which should
            belong the created process group
    Returns:
        New process group
    """
    new_group = dist.new_group(ranks=ranks)
    init_tensor_fp32, init_tensor_fp16 = (torch.zeros(1), torch.zeros(1).half())
    for init_tensor in [init_tensor_fp32, init_tensor_fp16]:
        if torch.cuda.is_available():
            init_tensor = init_tensor.cuda()
        if dist.get_rank() in ranks:
            dist.all_reduce(init_tensor, group=new_group)
        torch.cuda.synchronize()
    return new_group
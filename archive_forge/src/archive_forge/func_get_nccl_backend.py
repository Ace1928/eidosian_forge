import contextlib
import torch
from torch.distributed import ReduceOp
def get_nccl_backend():
    return _NCCL_BACKEND
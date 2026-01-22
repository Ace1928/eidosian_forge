import torch
import torch.distributed as dist
from torch.autograd import Variable
from dataclasses import dataclass
from typing import Any, no_type_check
from torch.distributed.utils import _free_storage

    Performs allreduce in the reduced precision given by DDP's mixed precision
    reduce_dtype, and upcasts parameters and gradients to fp32 in preparation
    to run the optimizer.
    
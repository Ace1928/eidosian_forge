import re
import sys
import time
from functools import partial, wraps
from typing import Tuple
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import _rref_context_get_debug_info
from torch.testing._internal.common_utils import FILE_SCHEMA, TEST_WITH_TSAN
def initialize_pg(init_method, rank: int, world_size: int) -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=world_size)
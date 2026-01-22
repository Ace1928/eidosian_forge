from typing import cast, Dict, List, Optional, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import prod
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
def merge_sharding(dim: str, a: int, b: int) -> int:
    if a != b:
        if a == -1 or b == -1:
            nonlocal needs_reshard
            needs_reshard = True
            return a if a != -1 else b
        else:
            raise RuntimeError(f'{equation}: dim {dim} sharded two different ways: {a} and {b}')
    else:
        return a
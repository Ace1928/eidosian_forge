from typing import Callable, List, Optional, Tuple
import torch
from .common import make_pytorch_cuda_operator
from .differentiable_collectives import (
from .sequence_parallel_fused_ops import (
from .tiled_matmul import tiled_matmul_fwd
def sequence_parallel_leading_matmul(x: torch.Tensor, ws: List[torch.Tensor], *, fuse: bool, process_group: torch.distributed.ProcessGroup) -> List[torch.Tensor]:
    os = _SequenceParallelLeadingMatmul.apply(fuse, process_group, x.flatten(0, -2), *ws)
    return [o.view(-1, *x.shape[1:-1], w.shape[1]) for o, w in zip(os, ws)]
from typing import Callable, List, Optional, Tuple
import torch
from .common import make_pytorch_cuda_operator
from .differentiable_collectives import (
from .sequence_parallel_fused_ops import (
from .tiled_matmul import tiled_matmul_fwd
def my_si_matmul(grad_gathered_inputs: List[torch.Tensor], dst_rank: int, stream_factory: Callable[[], torch.cuda.Stream]) -> None:
    grad_gi, = grad_gathered_inputs
    with torch.cuda.stream(stream_factory()):
        tiled_matmul_fwd([[grad_gos[dst_rank] for grad_gos in grad_gathered_outputss]], [[w.t()] for w in weights], out=[[grad_gi]])
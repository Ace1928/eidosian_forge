from typing import Callable, List, Optional, Tuple
import torch
from .common import make_pytorch_cuda_operator
from .differentiable_collectives import (
from .sequence_parallel_fused_ops import (
from .tiled_matmul import tiled_matmul_fwd
def my_gi_and_w_matmul(grad_gathered_outputs_shard: List[torch.Tensor], src_rank: int, stream_factory: Callable[[], torch.cuda.Stream]) -> None:
    grad_go_shard, = grad_gathered_outputs_shard
    with torch.cuda.stream(stream_factory()):
        torch.matmul(grad_go_shard, weight.t(), out=grad_gathered_inputs[src_rank])
    with torch.cuda.stream(stream_factory()):
        grad_weight.t().addmm_(grad_go_shard.t(), gathered_inputs[src_rank])
from typing import Callable, List, Optional, Tuple
import torch
from .common import make_pytorch_cuda_operator
from .differentiable_collectives import (
from .sequence_parallel_fused_ops import (
from .tiled_matmul import tiled_matmul_fwd
def my_w_matmul(gathered_inputs_shard: List[torch.Tensor], src_rank: int, stream_factory: Callable[[], torch.cuda.Stream]) -> None:
    gi_shard, = gathered_inputs_shard
    for grad_gos, grad_w, event in zip(grad_gathered_outputss, grad_weights, events):
        with torch.cuda.stream(stream_factory()):
            event.wait()
            grad_w.t().addmm_(grad_gos[src_rank].t(), gi_shard)
            event.record()
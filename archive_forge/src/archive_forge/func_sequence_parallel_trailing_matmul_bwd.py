from typing import Callable, List, Optional, Tuple
import torch
from .common import make_pytorch_cuda_operator
from .differentiable_collectives import (
from .sequence_parallel_fused_ops import (
from .tiled_matmul import tiled_matmul_fwd
@make_pytorch_cuda_operator
def sequence_parallel_trailing_matmul_bwd(gathered_input: torch.Tensor, weight: torch.Tensor, grad_scattered_output: torch.Tensor, fuse: bool, process_group: torch.distributed.ProcessGroup) -> Tuple[torch.Tensor, torch.Tensor]:
    mp_size = process_group.size()
    if fuse:
        grad_gathered_input = torch.empty_like(gathered_input)
        grad_weight = torch.zeros_like(weight)
        gathered_inputs = gathered_input.tensor_split(mp_size, dim=0)
        grad_gathered_inputs = grad_gathered_input.tensor_split(mp_size, dim=0)

        def my_gi_and_w_matmul(grad_gathered_outputs_shard: List[torch.Tensor], src_rank: int, stream_factory: Callable[[], torch.cuda.Stream]) -> None:
            grad_go_shard, = grad_gathered_outputs_shard
            with torch.cuda.stream(stream_factory()):
                torch.matmul(grad_go_shard, weight.t(), out=grad_gathered_inputs[src_rank])
            with torch.cuda.stream(stream_factory()):
                grad_weight.t().addmm_(grad_go_shard.t(), gathered_inputs[src_rank])
        fused_allgather_and_anything([grad_scattered_output], my_gi_and_w_matmul, group=process_group)
    else:
        grad_gathered_output = gather_along_first_dim(grad_scattered_output, process_group=process_group)
        grad_gathered_input = torch.matmul(grad_gathered_output, weight.t())
        grad_weight = torch.matmul(grad_gathered_output.t(), gathered_input).t()
    return (grad_gathered_input, grad_weight)
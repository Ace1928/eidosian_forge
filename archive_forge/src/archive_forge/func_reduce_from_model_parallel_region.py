from typing import Optional, Tuple
import torch
import torch.distributed
def reduce_from_model_parallel_region(x: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(x, process_group)
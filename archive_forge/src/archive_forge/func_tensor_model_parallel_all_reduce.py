from collections import namedtuple
from typing import Any, Dict, List, Optional, Union
import torch
from torch.distributed import ProcessGroup
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.custom_all_reduce import custom_all_reduce
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation will be applied in-place on the input tensor if
    disable_custom_all_reduce is set to True. Otherwise, this operation may or
    may not be applied in place depending on whether custom all reduce is
    invoked for a particular tensor, which further depends on the tensor size
    and GPU topology.

    TLDR: always assume this function modifies its input, but use the return
    value as the output. 
    """
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    out = custom_all_reduce(input_)
    if out is not None:
        return out
    if is_cupy_nccl_enabled_for_all_reduce():
        cupy_utils.all_reduce(input_)
    else:
        torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())
    return input_
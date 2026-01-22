import contextlib
import torch
from vllm.model_executor.parallel_utils import cupy_utils
def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())
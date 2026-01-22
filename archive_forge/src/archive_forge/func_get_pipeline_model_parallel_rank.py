import contextlib
import torch
from vllm.model_executor.parallel_utils import cupy_utils
def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())
import contextlib
import torch
from vllm.model_executor.parallel_utils import cupy_utils
def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP
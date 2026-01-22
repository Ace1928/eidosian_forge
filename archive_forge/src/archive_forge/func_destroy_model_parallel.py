import contextlib
import torch
from vllm.model_executor.parallel_utils import cupy_utils
def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    if _TENSOR_MODEL_PARALLEL_GROUP:
        torch.distributed.destroy_process_group(_TENSOR_MODEL_PARALLEL_GROUP)
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    if _PIPELINE_MODEL_PARALLEL_GROUP:
        torch.distributed.destroy_process_group(_PIPELINE_MODEL_PARALLEL_GROUP)
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = None
    cupy_utils.destroy_process_group()
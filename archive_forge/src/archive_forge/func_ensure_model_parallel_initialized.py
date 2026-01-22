import contextlib
import torch
from vllm.model_executor.parallel_utils import cupy_utils
def ensure_model_parallel_initialized(tensor_model_parallel_size: int, pipeline_model_parallel_size: int) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)
        return
    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size, f'tensor parallel group already initialized, but of unexpected size: get_tensor_model_parallel_world_size()={get_tensor_model_parallel_world_size()!r} vs. tensor_model_parallel_size={tensor_model_parallel_size!r}'
    assert get_pipeline_model_parallel_world_size() == pipeline_model_parallel_size, f'pipeline parallel group already initialized, but of unexpected size: get_pipeline_model_parallel_world_size()={get_pipeline_model_parallel_world_size()!r} vs. pipeline_model_parallel_size={pipeline_model_parallel_size!r}'
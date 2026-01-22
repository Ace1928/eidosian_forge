import contextlib
import torch
from vllm.model_executor.parallel_utils import cupy_utils
def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, 'Pipeline parallel group is not initialized'
    return _PIPELINE_GLOBAL_RANKS[0]
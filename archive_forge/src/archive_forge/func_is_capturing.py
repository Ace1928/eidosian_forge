from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
def is_capturing() -> bool:
    return _IS_CAPTURING and _CA_HANDLE is not None
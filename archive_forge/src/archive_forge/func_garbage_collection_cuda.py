import gc
from typing import Any
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
def garbage_collection_cuda() -> None:
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError as exception:
        if not is_oom_error(exception):
            raise
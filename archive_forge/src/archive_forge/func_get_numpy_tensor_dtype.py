import asyncio
import time
from typing import List
import numpy
import ray
import ray.experimental.internal_kv as internal_kv
from ray._raylet import GcsClient
from ray.util.collective.types import ReduceOp, torch_available
from ray.util.queue import _QueueActor
def get_numpy_tensor_dtype(tensor):
    """Return the corresponded Cupy dtype given a tensor."""
    if isinstance(tensor, numpy.ndarray):
        return tensor.dtype.type
    if torch_available():
        if isinstance(tensor, torch.Tensor):
            return TORCH_NUMPY_DTYPE_MAP[tensor.dtype]
    raise ValueError('Unsupported tensor type. Got: {}. Supported CPU tensor types are: torch.Tensor, numpy.ndarray.'.format(type(tensor)))
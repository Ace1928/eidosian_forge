import numpy
from ray.util.collective.types import ReduceOp, torch_available
Returns the gpu devices of the list of input tensors.

    Args:
        tensors: a list of tensors, each locates on a GPU.

    Returns:
        list: the list of GPU devices.

    
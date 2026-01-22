import numpy
from ray.util.collective.types import ReduceOp, torch_available
def get_tensor_strides(tensor):
    """Return the strides of the tensor as a list."""
    if isinstance(tensor, cupy.ndarray):
        return [int(stride / tensor.dtype.itemsize) for stride in tensor.strides]
    if torch_available():
        if isinstance(tensor, torch.Tensor):
            return list(tensor.stride())
    raise ValueError('Unsupported tensor type. Got: {}. Supported GPU tensor types are: torch.Tensor, cupy.ndarray.'.format(type(tensor)))
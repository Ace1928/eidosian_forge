import numpy
from ray.util.collective.types import ReduceOp, torch_available
def get_tensor_ptr(tensor):
    """Return the pointer to the underlying memory storage of a tensor."""
    if isinstance(tensor, cupy.ndarray):
        return tensor.data.ptr
    if isinstance(tensor, numpy.ndarray):
        return tensor.data
    if torch_available():
        if isinstance(tensor, torch.Tensor):
            if not tensor.is_cuda:
                raise RuntimeError('Torch tensor must be on GPU when using NCCL collectives.')
            return tensor.data_ptr()
    raise ValueError('Unsupported tensor type. Got: {}. Supported GPU tensor types are: torch.Tensor, cupy.ndarray.'.format(type(tensor)))
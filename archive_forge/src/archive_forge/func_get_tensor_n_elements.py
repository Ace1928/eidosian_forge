import numpy
from ray.util.collective.types import ReduceOp, torch_available
def get_tensor_n_elements(tensor):
    """Return the number of elements in a tensor."""
    if isinstance(tensor, cupy.ndarray) or isinstance(tensor, numpy.ndarray):
        return tensor.size
    if torch_available():
        if isinstance(tensor, torch.Tensor):
            return torch.numel(tensor)
    raise ValueError('Unsupported tensor type. Got: {}. Supported GPU tensor types are: torch.Tensor, cupy.ndarray.'.format(type(tensor)))
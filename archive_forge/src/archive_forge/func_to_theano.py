import numpy as np
from ..sharing import to_backend_cache_wrap
@to_backend_cache_wrap(constants=True)
def to_theano(array, constant=False):
    """Convert a numpy array to ``theano.tensor.TensorType`` instance.
    """
    import theano
    if isinstance(array, np.ndarray):
        if constant:
            return theano.tensor.constant(array)
        return theano.tensor.TensorType(dtype=array.dtype, broadcastable=[False] * len(array.shape))()
    return array
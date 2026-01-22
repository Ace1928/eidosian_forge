import numpy as np
from ..sharing import to_backend_cache_wrap
@to_backend_cache_wrap(constants=True)
def to_tensorflow(array, constant=False):
    """Convert a numpy array to a ``tensorflow.placeholder`` instance.
    """
    tf, device, eager = _get_tensorflow_and_device()
    if eager:
        if isinstance(array, np.ndarray):
            with tf.device(device):
                return tf.convert_to_tensor(array)
        return array
    if isinstance(array, np.ndarray):
        if constant:
            return tf.convert_to_tensor(array)
        return tf.placeholder(array.dtype, array.shape)
    return array
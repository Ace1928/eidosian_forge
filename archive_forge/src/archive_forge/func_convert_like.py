import warnings
import autoray as ar
import numpy as _np
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from . import single_dispatch  # pylint:disable=unused-import
def convert_like(tensor1, tensor2):
    """Convert a tensor to the same type as another.

    Args:
        tensor1 (tensor_like): tensor to convert
        tensor2 (tensor_like): tensor with corresponding type to convert to

    Returns:
        tensor_like: a tensor with the same shape, values, and dtype as ``tensor1`` and the
        same type as ``tensor2``.

    **Example**

    >>> x = np.array([1, 2])
    >>> y = tf.Variable([3, 4])
    >>> convert_like(x, y)
    <tf.Tensor: shape=(2,), dtype=int64, numpy=array([1, 2])>
    """
    interface = get_interface(tensor2)
    if interface == 'torch':
        dev = tensor2.device
        return np.asarray(tensor1, device=dev, like=interface)
    return np.asarray(tensor1, like=interface)
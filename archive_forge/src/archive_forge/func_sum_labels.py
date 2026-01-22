import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def sum_labels(input, labels=None, index=None):
    """Calculates the sum of the values of an n-D image array, optionally
       at specified sub-regions.

    Args:
        input (cupy.ndarray): Nd-image data to process.
        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.
            If not None, must be same shape as `input`.
        index (cupy.ndarray or None): `labels` to include in output. If None
            (default), all values where `labels` is non-zero are used.

    Returns:
       sum (cupy.ndarray): sum of values, for each sub-region if
       `labels` and `index` are specified.

    .. seealso:: :func:`scipy.ndimage.sum_labels`
    """
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')
    if input.dtype in (cupy.complex64, cupy.complex128):
        raise TypeError('cupyx.scipy.ndimage.sum does not support %{}'.format(input.dtype.type))
    use_kern = False
    if input.dtype not in [cupy.int32, cupy.float16, cupy.float32, cupy.float64, cupy.uint32, cupy.uint64, cupy.ulonglong]:
        warnings.warn('Using the slower implmentation as cupyx.scipy.ndimage.sum supports int32, float16, float32, float64, uint32, uint64 as data typesfor the fast implmentation', _util.PerformanceWarning)
        use_kern = True
    if labels is None:
        return input.sum()
    if not isinstance(labels, cupy.ndarray):
        raise TypeError('label must be cupy.ndarray')
    input, labels = cupy.broadcast_arrays(input, labels)
    if index is None:
        return input[labels != 0].sum()
    if not isinstance(index, cupy.ndarray):
        if not isinstance(index, int):
            raise TypeError('index must be cupy.ndarray or a scalar int')
        else:
            return input[labels == index].sum()
    if index.size == 0:
        return cupy.array([], dtype=cupy.int64)
    out = cupy.zeros_like(index, dtype=cupy.float64)
    if input.size >= 262144 and index.size <= 4 or use_kern:
        return _ndimage_sum_kernel_2(input, labels, index, out)
    return _ndimage_sum_kernel(input, labels, index, index.size, out)
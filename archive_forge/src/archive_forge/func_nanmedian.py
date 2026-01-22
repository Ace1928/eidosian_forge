import functools
import numpy
import cupy
from cupy._core import _routines_statistics as _statistics
def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Compute the median along the specified axis, while ignoring NaNs.

    Returns the median of the array elements.

    Args:
        a (cupy.ndarray): Array to compute the median.
        axis (int, sequence of int or None): Axis along which the medians are
            computed. The flattened array is used by default.
        out (cupy.ndarray): Output array.
        overwrite_input (bool): If ``True``, then allow use of memory of input
            array a for calculations. The input array will be modified by the
            call to median. This will save memory when you do not need to
            preserve the contents of the input array. Treat the input as
            undefined, but it will probably be fully or partially sorted.
            Default is ``False``. If ``overwrite_input`` is ``True`` and ``a``
            is not already an ndarray, an error will be raised.
        keepdims (bool): If ``True``, the axis is remained as an axis of size
            one.

    Returns:
        cupy.ndarray: The median of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.nanmedian`

    """
    if a.dtype.char in 'efdFD':
        return _statistics._nanmedian(a, axis, out, overwrite_input, keepdims)
    else:
        return median(a, axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims)
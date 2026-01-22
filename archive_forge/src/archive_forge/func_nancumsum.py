import numpy
import cupy
from cupy._core import _routines_math as _math
from cupy._core import _fusion_thread_local
from cupy._core import internal
def nancumsum(a, axis=None, dtype=None, out=None):
    """Returns the cumulative sum of an array along a given axis treating Not a
    Numbers (NaNs) as zero.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative sum is taken. If it is not
            specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.nancumsum`
    """
    a = _replace_nan(a, 0, out=out)
    return cumsum(a, axis=axis, dtype=dtype, out=out)
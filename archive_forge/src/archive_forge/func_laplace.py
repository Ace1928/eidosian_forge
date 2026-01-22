import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def laplace(input, output=None, mode='reflect', cval=0.0):
    """Multi-dimensional Laplace filter based on approximate second
    derivatives.

    Args:
        input (cupy.ndarray): The input array.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.laplace`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights_dtype = _util._init_weights_dtype(input)
    weights = cupy.array([1, -2, 1], dtype=weights_dtype)

    def derivative2(input, axis, output, mode, cval):
        return correlate1d(input, weights, axis, output, mode, cval)
    return generic_laplace(input, derivative2, output, mode, cval)
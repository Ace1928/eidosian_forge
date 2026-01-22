import math
import warnings
import cupy
import numpy
from cupy import _core
from cupy._core import internal
from cupy.cuda import runtime
from cupyx import _texture
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _interp_kernels
from cupyx.scipy.ndimage import _spline_prefilter_core
def spline_filter(input, order=3, output=cupy.float64, mode='mirror'):
    """Multidimensional spline filter.

    Args:
        input (cupy.ndarray): The input array.
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        output (cupy.ndarray or dtype, optional): The array in which to place
            the output, or the dtype of the returned array. Default is
            ``numpy.float64``.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).

    Returns:
        cupy.ndarray: The result of prefiltering the input.

    .. seealso:: :func:`scipy.spline_filter1d`
    """
    if order < 2 or order > 5:
        raise RuntimeError('spline order not supported')
    x = input
    temp, data_dtype, output_dtype = _get_spline_output(x, output)
    if order not in [0, 1] and input.ndim > 0:
        for axis in range(x.ndim):
            spline_filter1d(x, order, axis, output=temp, mode=mode)
            x = temp
    if isinstance(output, cupy.ndarray):
        _core.elementwise_copy(temp, output)
    else:
        output = temp
    if output.dtype != output_dtype:
        output = output.astype(output_dtype)
    return output
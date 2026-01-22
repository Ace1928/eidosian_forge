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
def shift(input, shift, output=None, order=3, mode='constant', cval=0.0, prefilter=True):
    """Shift an array.

    The array is shifted using spline interpolation of the requested order.
    Points outside the boundaries of the input are filled according to the
    given mode.

    Args:
        input (cupy.ndarray): The input array.
        shift (float or sequence): The shift along the axes. If a float,
            ``shift`` is the same for each axis. If a sequence, ``shift``
            should contain one value for each axis.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array.
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).
        cval (scalar): Value used for points outside the boundaries of
            the input if ``mode='constant'`` or ``mode='opencv'``. Default is
            0.0
        prefilter (bool): Determines if the input array is prefiltered with
            ``spline_filter`` before interpolation. The default is True, which
            will create a temporary ``float64`` array of filtered values if
            ``order > 1``. If setting this to False, the output will be
            slightly blurred if ``order > 1``, unless the input is prefiltered,
            i.e. it is the result of calling ``spline_filter`` on the original
            input.

    Returns:
        cupy.ndarray or None:
            The shifted input.

    .. seealso:: :func:`scipy.ndimage.shift`
    """
    _check_parameter('shift', order, mode)
    shift = _util._fix_sequence_arg(shift, input.ndim, 'shift', float)
    if mode == 'opencv':
        mode = '_opencv_edge'
        output = affine_transform(input, cupy.ones(input.ndim, input.dtype), cupy.negative(cupy.asarray(shift)), None, output, order, mode, cval, prefilter)
    else:
        output = _util._get_output(output, input)
        if input.dtype.kind in 'iu':
            input = input.astype(cupy.float32)
        filtered, nprepad = _filter_input(input, prefilter, mode, cval, order)
        integer_output = output.dtype.kind in 'iu'
        _util._check_cval(mode, cval, integer_output)
        large_int = _prod(input.shape) > 1 << 31
        kern = _interp_kernels._get_shift_kernel(input.ndim, large_int, input.shape, mode, cval=cval, order=order, integer_output=integer_output, nprepad=nprepad)
        shift = cupy.asarray(shift, dtype=cupy.float64, order='C')
        if shift.ndim != 1:
            raise ValueError('shift must be 1d')
        if shift.size != filtered.ndim:
            raise ValueError('len(shift) must equal input.ndim')
        kern(filtered, shift, output)
    return output
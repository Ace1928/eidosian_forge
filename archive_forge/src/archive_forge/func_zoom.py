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
def zoom(input, zoom, output=None, order=3, mode='constant', cval=0.0, prefilter=True, *, grid_mode=False):
    """Zoom an array.

    The array is zoomed using spline interpolation of the requested order.

    Args:
        input (cupy.ndarray): The input array.
        zoom (float or sequence): The zoom factor along the axes. If a float,
            ``zoom`` is the same for each axis. If a sequence, ``zoom`` should
            contain one value for each axis.
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
        grid_mode (bool, optional): If False, the distance from the pixel
            centers is zoomed. Otherwise, the distance including the full pixel
            extent is used. For example, a 1d signal of length 5 is considered
            to have length 4 when ``grid_mode`` is False, but length 5 when
            ``grid_mode`` is True. See the following visual illustration:

            .. code-block:: text

                    | pixel 1 | pixel 2 | pixel 3 | pixel 4 | pixel 5 |
                         |<-------------------------------------->|
                                            vs.
                    |<----------------------------------------------->|

            The starting point of the arrow in the diagram above corresponds to
            coordinate location 0 in each mode.

    Returns:
        cupy.ndarray or None:
            The zoomed input.

    .. seealso:: :func:`scipy.ndimage.zoom`
    """
    _check_parameter('zoom', order, mode)
    zoom = _util._fix_sequence_arg(zoom, input.ndim, 'zoom', float)
    output_shape = []
    for s, z in zip(input.shape, zoom):
        output_shape.append(int(round(s * z)))
    output_shape = tuple(output_shape)
    if mode == 'opencv':
        zoom = []
        offset = []
        for in_size, out_size in zip(input.shape, output_shape):
            if out_size > 0:
                zoom.append(float(in_size) / out_size)
                offset.append((zoom[-1] - 1) / 2.0)
            else:
                zoom.append(0)
                offset.append(0)
        mode = 'nearest'
        output = affine_transform(input, cupy.asarray(zoom), offset, output_shape, output, order, mode, cval, prefilter)
    else:
        if grid_mode:
            suggest_mode = None
            if mode == 'constant':
                suggest_mode = 'grid-constant'
            elif mode == 'wrap':
                suggest_mode = 'grid-wrap'
            if suggest_mode is not None:
                warnings.warn(f'It is recommended to use mode = {suggest_mode} instead of {mode} when grid_mode is True.')
        zoom = []
        for in_size, out_size in zip(input.shape, output_shape):
            if grid_mode and out_size > 0:
                zoom.append(in_size / out_size)
            elif out_size > 1:
                zoom.append((in_size - 1) / (out_size - 1))
            else:
                zoom.append(0)
        output = _util._get_output(output, input, shape=output_shape)
        if input.dtype.kind in 'iu':
            input = input.astype(cupy.float32)
        filtered, nprepad = _filter_input(input, prefilter, mode, cval, order)
        integer_output = output.dtype.kind in 'iu'
        _util._check_cval(mode, cval, integer_output)
        large_int = max(_prod(input.shape), _prod(output_shape)) > 1 << 31
        kern = _interp_kernels._get_zoom_kernel(input.ndim, large_int, output_shape, mode, order=order, integer_output=integer_output, grid_mode=grid_mode, nprepad=nprepad)
        zoom = cupy.asarray(zoom, dtype=cupy.float64)
        kern(filtered, zoom, output)
    return output
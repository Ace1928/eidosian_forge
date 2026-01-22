import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def generic_filter1d(input, function, filter_size, axis=-1, output=None, mode='reflect', cval=0.0, origin=0):
    """Compute a 1D filter along the given axis using the provided raw kernel.

    Unlike the scipy.ndimage function, this does not support the
    ``extra_arguments`` or ``extra_keywordsdict`` arguments and has significant
    restrictions on the ``function`` provided.

    Args:
        input (cupy.ndarray): The input array.
        function (cupy.RawKernel): The kernel to apply along each axis.
        filter_size (int): Length of the filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. note::
        The provided function (as a RawKernel) must have the following
        signature. Unlike most functions, this should not utilize
        `blockDim`/`blockIdx`/`threadIdx`::

            __global__ void func(double *input_line, ptrdiff_t input_length,
                                 double *output_line, ptrdiff_t output_length)

    .. seealso:: :func:`scipy.ndimage.generic_filter1d`
    """
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported')
    if not isinstance(function, cupy.RawKernel):
        raise TypeError('bad function type')
    if filter_size < 1:
        raise RuntimeError('invalid filter size')
    axis = internal._normalize_axis_index(axis, input.ndim)
    origin = _util._check_origin(origin, filter_size)
    _util._check_mode(mode)
    output = _util._get_output(output, input)
    in_ctype = cupy._core._scalar.get_typename(input.dtype)
    out_ctype = cupy._core._scalar.get_typename(output.dtype)
    int_type = _util._get_inttype(input)
    n_lines = input.size // input.shape[axis]
    kernel = _filters_generic._get_generic_filter1d(function, input.shape[axis], n_lines, filter_size, origin, mode, float(cval), in_ctype, out_ctype, int_type)
    data = cupy.array((axis, input.ndim) + input.shape + input.strides + output.strides, dtype=cupy.int32 if int_type == 'int' else cupy.int64)
    kernel(((n_lines + 128 - 1) // 128,), (128,), (input, output, data))
    return output
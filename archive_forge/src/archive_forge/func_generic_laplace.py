import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def generic_laplace(input, derivative2, output=None, mode='reflect', cval=0.0, extra_arguments=(), extra_keywords=None):
    """Multi-dimensional Laplace filter using a provided second derivative
    function.

    Args:
        input (cupy.ndarray): The input array.
        derivative2 (callable): Function or other callable with the following
            signature that is called once per axis::

                derivative2(input, axis, output, mode, cval,
                            *extra_arguments, **extra_keywords)

            where ``input`` and ``output`` are ``cupy.ndarray``, ``axis`` is an
            ``int`` from ``0`` to the number of dimensions, and ``mode``,
            ``cval``, ``extra_arguments``, ``extra_keywords`` are the values
            given to this function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        extra_arguments (sequence, optional):
            Sequence of extra positional arguments to pass to ``derivative2``.
        extra_keywords (dict, optional):
            dict of extra keyword arguments to pass ``derivative2``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.generic_laplace`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    if extra_keywords is None:
        extra_keywords = {}
    ndim = input.ndim
    modes = _util._fix_sequence_arg(mode, ndim, 'mode', _util._check_mode)
    output = _util._get_output(output, input)
    if ndim == 0:
        _core.elementwise_copy(input, output)
        return output
    derivative2(input, 0, output, modes[0], cval, *extra_arguments, **extra_keywords)
    if ndim > 1:
        tmp = _util._get_output(output.dtype, input)
        for i in range(1, ndim):
            derivative2(input, i, tmp, modes[i], cval, *extra_arguments, **extra_keywords)
            output += tmp
    return output
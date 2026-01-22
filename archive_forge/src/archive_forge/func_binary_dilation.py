import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
def binary_dilation(input, structure=None, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False):
    """Multidimensional binary dilation with the given structuring element.

    Args:
        input(cupy.ndarray): The input binary array_like to be dilated.
            Non-zero (True) elements form the subset to be dilated.
        structure(cupy.ndarray, optional): The structuring element used for the
            dilation. Non-zero elements are considered True. If no structuring
            element is provided an element is generated with a square
            connectivity equal to one. (Default value = None).
        iterations(int, optional): The dilation is repeated ``iterations``
            times (one, by default). If iterations is less than 1, the dilation
            is repeated until the result does not change anymore. Only an
            integer of iterations is accepted.
        mask(cupy.ndarray or None, optional): If a mask is given, only those
            elements with a True value at the corresponding mask element are
            modified at each iteration. (Default value = None)
        output(cupy.ndarray, optional): Array of the same shape as input, into
            which the output is placed. By default, a new array is created.
        border_value(int (cast to 0 or 1), optional): Value at the
            border in the output array. (Default value = 0)
        origin(int or tuple of ints, optional): Placement of the filter, by
            default 0.
        brute_force(boolean, optional): Memory condition: if False, only the
            pixels whose value was changed in the last iteration are tracked as
            candidates to be updated (dilated) in the current iteration; if
            True all pixels are considered as candidates for dilation,
            regardless of what happened in the previous iteration.

    Returns:
        cupy.ndarray: The result of binary dilation.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_dilation`
    """
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    origin = _util._fix_sequence_arg(origin, input.ndim, 'origin', int)
    structure = structure[tuple([slice(None, None, -1)] * structure.ndim)]
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1
    return _binary_erosion(input, structure, iterations, mask, output, border_value, origin, 1, brute_force)
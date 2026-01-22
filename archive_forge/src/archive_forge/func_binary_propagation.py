import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
def binary_propagation(input, structure=None, mask=None, output=None, border_value=0, origin=0):
    """
    Multidimensional binary propagation with the given structuring element.

    Args:
        input (cupy.ndarray): Binary image to be propagated inside ``mask``.
        structure (cupy.ndarray, optional): Structuring element used in the
            successive dilations. The output may depend on the structuring
            element, especially if ``mask`` has several connex components. If
            no structuring element is provided, an element is generated with a
            squared connectivity equal to one.
        mask (cupy.ndarray, optional): Binary mask defining the region into
            which ``input`` is allowed to propagate.
        output (cupy.ndarray, optional): Array of the same shape as input, into
            which the output is placed. By default, a new array is created.
        border_value (int, optional): Value at the border in the output array.
            The value is cast to 0 or 1.
        origin (int or tuple of ints, optional): Placement of the filter.

    Returns:
        cupy.ndarray : Binary propagation of ``input`` inside ``mask``.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_propagation`
    """
    return binary_dilation(input, structure, -1, mask, output, border_value, origin, brute_force=True)
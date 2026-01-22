import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def median(input, labels=None, index=None):
    """Calculate the median of the values of an array over labeled regions.

    Args:
        input (cupy.ndarray):
            Array of values. For each region specified by `labels`, the
            median values of `input` over the region is computed.
        labels (cupy.ndarray, optional): An array of integers marking different
            regions over which the median value of `input` is to be computed.
            `labels` must have the same shape as `input`. If `labels` is not
            specified, the median over the whole array is returned.
        index (array_like, optional): A list of region labels that are taken
            into account for computing the medians. If `index` is None, the
            median over all elements where `labels` is non-zero is returned.

    Returns:
        cupy.ndarray: Array of medians of `input` over the regions
        determined by `labels` and whose index is in `index`. If `index` or
        `labels` are not specified, a 0-dimensional cupy.ndarray is
        returned: the median value of `input` if `labels` is None,
        and the median value of elements where `labels` is greater than
        zero if `index` is None.

    .. seealso:: :func:`scipy.ndimage.median`
    """
    return _select(input, labels, index, find_median=True)[0]
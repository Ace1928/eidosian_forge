import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import common
from cupy.cuda import runtime
def histogram2d(x, y, bins=10, range=None, weights=None, density=None):
    """Compute the bi-dimensional histogram of two data samples.

    Args:
        x (cupy.ndarray): The first array of samples to be histogrammed.
        y (cupy.ndarray): The second array of samples to be histogrammed.
        bins (int or tuple of int or cupy.ndarray): The bin specification:

            * A sequence of arrays describing the monotonically increasing bin
              edges along each dimension.
            * The number of bins for each dimension (nx, ny)
            * The number of bins for all dimensions (nx=ny=bins).
        range (sequence, optional): A sequence of length two, each an optional
            (lower, upper) tuple giving the outer bin edges to be used if the
            edges are not given explicitly in `bins`. An entry of None in the
            sequence results in the minimum and maximum values being used for
            the corresponding dimension. The default, None, is equivalent to
            passing a tuple of two None values.
        weights (cupy.ndarray): An array of values `w_i` weighing each sample
            `(x_i, y_i)`. The values of the returned histogram are equal to the
            sum of the weights belonging to the samples falling into each bin.
        density (bool, optional): If False, the default, returns the number of
            samples in each bin. If True, returns the probability *density*
            function at the bin, ``bin_count / sample_count / bin_volume``.

    Returns:
        tuple:
        H (cupy.ndarray):
            The multidimensional histogram of sample x. See
            normed and weights for the different possible semantics.
        edges0 (tuple of cupy.ndarray):
            A list of D arrays describing the bin
            edges for the first dimension.
        edges1 (tuple of cupy.ndarray):
            A list of D arrays describing the bin
            edges for the second dimension.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.histogram2d`
    """
    try:
        n = len(bins)
    except TypeError:
        n = 1
    if n != 1 and n != 2:
        if isinstance(bins, cupy.ndarray):
            xedges = yedges = bins
            bins = [xedges, yedges]
        else:
            raise ValueError('array-like bins not supported in CuPy')
    hist, edges = histogramdd([x, y], bins, range, weights, density)
    return (hist, edges[0], edges[1])
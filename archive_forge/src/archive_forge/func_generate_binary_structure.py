import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
def generate_binary_structure(rank, connectivity):
    """Generate a binary structure for binary morphological operations.

    Args:
        rank(int): Number of dimensions of the array to which the structuring
            element will be applied, as returned by ``np.ndim``.
        connectivity(int): ``connectivity`` determines which elements of the
            output array belong to the structure, i.e., are considered as
            neighbors of the central element. Elements up to a squared distance
            of ``connectivity`` from the center are considered neighbors.
            ``connectivity`` may range from 1 (no diagonal elements are
            neighbors) to ``rank`` (all elements are neighbors).

    Returns:
        cupy.ndarray: Structuring element which may be used for binary
        morphological operations, with ``rank`` dimensions and all
        dimensions equal to 3.

    .. seealso:: :func:`scipy.ndimage.generate_binary_structure`
    """
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return cupy.asarray(True, dtype=bool)
    output = numpy.fabs(numpy.indices([3] * rank) - 1)
    output = numpy.add.reduce(output, 0)
    output = output <= connectivity
    return cupy.asarray(output)
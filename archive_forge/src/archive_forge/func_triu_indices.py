import functools
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy._creation import from_data
from cupy._manipulation import join
def triu_indices(n, k=0, m=None):
    """Returns the indices of the upper triangular matrix.
    Here, the first group of elements contains row coordinates
    of all indices and the second group of elements
    contains column coordinates.

    Parameters
    ----------
    n : int
        The size of the arrays for which the returned indices will
        be valid.
    k : int, optional
        Refers to the diagonal offset. By default, `k = 0` i.e.
        the main dialogal. The positive value of `k`
        denotes the diagonals above the main diagonal, while the negative
        value includes the diagonals below the main diagonal.
    m : int, optional
        The column dimension of the arrays for which the
        returned arrays will be valid. By default, `m = n`.

    Returns
    -------
    y : tuple of ndarrays
        The indices for the triangle. The returned tuple
        contains two arrays, each with the indices along
        one dimension of the array.

    See Also
    --------
    numpy.triu_indices

    """
    tri_ = ~cupy.tri(n, m, k=k - 1, dtype=bool)
    return tuple((cupy.broadcast_to(inds, tri_.shape)[tri_] for inds in cupy.indices(tri_.shape, dtype=int)))
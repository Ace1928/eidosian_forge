import itertools
import warnings
from . import core as ma
from .core import (
import numpy as np
from numpy import ndarray, array as nxarray
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
from numpy.lib.index_tricks import AxisConcatenator
def notmasked_edges(a, axis=None):
    """
    Find the indices of the first and last unmasked values along an axis.

    If all values are masked, return None.  Otherwise, return a list
    of two tuples, corresponding to the indices of the first and last
    unmasked values respectively.

    Parameters
    ----------
    a : array_like
        The input array.
    axis : int, optional
        Axis along which to perform the operation.
        If None (default), applies to a flattened version of the array.

    Returns
    -------
    edges : ndarray or list
        An array of start and end indexes if there are any masked data in
        the array. If there are no masked data in the array, `edges` is a
        list of the first and last index.

    See Also
    --------
    flatnotmasked_contiguous, flatnotmasked_edges, notmasked_contiguous
    clump_masked, clump_unmasked

    Examples
    --------
    >>> a = np.arange(9).reshape((3, 3))
    >>> m = np.zeros_like(a)
    >>> m[1:, 1:] = 1

    >>> am = np.ma.array(a, mask=m)
    >>> np.array(am[~am.mask])
    array([0, 1, 2, 3, 6])

    >>> np.ma.notmasked_edges(am)
    array([0, 6])

    """
    a = asarray(a)
    if axis is None or a.ndim == 1:
        return flatnotmasked_edges(a)
    m = getmaskarray(a)
    idx = array(np.indices(a.shape), mask=np.asarray([m] * a.ndim))
    return [tuple([idx[i].min(axis).compressed() for i in range(a.ndim)]), tuple([idx[i].max(axis).compressed() for i in range(a.ndim)])]
import functools
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy._creation import from_data
from cupy._manipulation import join
def triu_indices_from(arr, k=0):
    """Returns indices for the upper-triangle of arr.

    Parameters
    ----------
    arr : cupy.ndarray
          The indices are valid for square arrays.
    k : int, optional
        Diagonal offset (see 'triu_indices` for details).

    Returns
    -------
    triu_indices_from : tuple of ndarrays
        Indices for the upper-triangle of `arr`.

    See Also
    --------
    numpy.triu_indices_from

    """
    if arr.ndim != 2:
        raise ValueError('input array must be 2-d')
    return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])
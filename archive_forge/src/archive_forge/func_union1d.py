import cupy
from cupy._core import _routines_logic as _logic
from cupy._core import _fusion_thread_local
from cupy._sorting import search as _search
from cupy import _util
def union1d(arr1, arr2):
    """Find the union of two arrays.

    Returns the unique, sorted array of values that are in either of
    the two input arrays.

    Parameters
    ----------
    arr1, arr2 : cupy.ndarray
        Input arrays. They are flattend if they are not already 1-D.

    Returns
    -------
    union1d : cupy.ndarray
        Sorted union of the input arrays.

    See Also
    --------
    numpy.union1d

    """
    return cupy.unique(cupy.concatenate((arr1, arr2), axis=None))
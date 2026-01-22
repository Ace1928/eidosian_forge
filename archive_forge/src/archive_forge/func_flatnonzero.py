import cupy
from cupy import _core
from cupy._core import fusion
from cupy import _util
from cupy._core import _routines_indexing as _indexing
from cupy._core import _routines_statistics as _statistics
def flatnonzero(a):
    """Return indices that are non-zero in the flattened version of a.

    This is equivalent to a.ravel().nonzero()[0].

    Args:
        a (cupy.ndarray): input array

    Returns:
        cupy.ndarray: Output array,
        containing the indices of the elements of a.ravel() that are non-zero.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.flatnonzero`
    """
    _util.check_array(a, arg_name='a')
    return a.ravel().nonzero()[0]
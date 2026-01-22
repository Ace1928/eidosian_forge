import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('block_diag')
def block_diag(*arrs):
    """Create a block diagonal matrix from provided arrays.

    Given the inputs ``A``, ``B``, and ``C``, the output will have these
    arrays arranged on the diagonal::

        [A, 0, 0]
        [0, B, 0]
        [0, 0, C]

    Args:
        A, B, C, ... (cupy.ndarray): Input arrays. A 1-D array of length ``n``
            is treated as a 2-D array with shape ``(1,n)``.

    Returns:
        (cupy.ndarray): Array with ``A``, ``B``, ``C``, ... on the diagonal.
        Output has the same dtype as ``A``.

    .. seealso:: :func:`scipy.linalg.block_diag`
    """
    if not arrs:
        return cupy.empty((1, 0))
    if len(arrs) == 1:
        arrs = (cupy.atleast_2d(*arrs),)
    else:
        arrs = cupy.atleast_2d(*arrs)
    if any((a.ndim != 2 for a in arrs)):
        bad = [k for k in range(len(arrs)) if arrs[k].ndim != 2]
        raise ValueError('arguments in the following positions have dimension greater than 2: {}'.format(bad))
    shapes = tuple((a.shape for a in arrs))
    shape = tuple((sum(x) for x in zip(*shapes)))
    dtype = cupy.find_common_type([a.dtype for a in arrs], [])
    out = cupy.zeros(shape, dtype=dtype)
    r, c = (0, 0)
    for arr in arrs:
        rr, cc = arr.shape
        out[r:r + rr, c:c + cc] = arr
        r += rr
        c += cc
    return out
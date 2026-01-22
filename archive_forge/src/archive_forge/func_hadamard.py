import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('hadamard')
def hadamard(n, dtype=int):
    """Construct an Hadamard matrix.

    Constructs an n-by-n Hadamard matrix, using Sylvester's construction. ``n``
    must be a power of 2.

    Args:
        n (int): The order of the matrix. ``n`` must be a power of 2.
        dtype (dtype, optional): The data type of the array to be constructed.

    Returns:
        (cupy.ndarray): The Hadamard matrix.

    .. seealso:: :func:`scipy.linalg.hadamard`
    """
    lg2 = 0 if n < 1 else int(n).bit_length() - 1
    if 2 ** lg2 != n:
        raise ValueError('n must be an positive a power of 2 integer')
    H = cupy.empty((n, n), dtype)
    return _hadamard_kernel(H, H)
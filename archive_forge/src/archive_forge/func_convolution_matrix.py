import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('convolution_matrix')
def convolution_matrix(a, n, mode='full'):
    """Construct a convolution matrix.

    Constructs the Toeplitz matrix representing one-dimensional convolution.

    Args:
        a (cupy.ndarray): The 1-D array to convolve.
        n (int): The number of columns in the resulting matrix. It gives the
            length of the input to be convolved with ``a``. This is analogous
            to the length of ``v`` in ``numpy.convolve(a, v)``.
        mode (str): This must be one of (``'full'``, ``'valid'``, ``'same'``).
            This is analogous to ``mode`` in ``numpy.convolve(v, a, mode)``.

    Returns:
        cupy.ndarray: The convolution matrix whose row count k depends on
        ``mode``:

        =========== =========================
        ``mode``    k
        =========== =========================
        ``'full'``  m + n - 1
        ``'same'``  max(m, n)
        ``'valid'`` max(m, n) - min(m, n) + 1
        =========== =========================

    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`
    .. seealso:: :func:`scipy.linalg.convolution_matrix`
    """
    if n <= 0:
        raise ValueError('n must be a positive integer.')
    if a.ndim != 1:
        raise ValueError('convolution_matrix expects a one-dimensional array as input')
    if a.size == 0:
        raise ValueError('len(a) must be at least 1.')
    if mode not in ('full', 'valid', 'same'):
        raise ValueError("`mode` argument must be one of ('full', 'valid', 'same')")
    az = cupy.pad(a, (0, n - 1), 'constant')
    raz = cupy.pad(a[::-1], (0, n - 1), 'constant')
    if mode == 'same':
        trim = min(n, a.size) - 1
        tb = trim // 2
        te = trim - tb
        col0 = az[tb:az.size - te]
        row0 = raz[-n - tb:raz.size - tb]
    elif mode == 'valid':
        tb = min(n, a.size) - 1
        te = tb
        col0 = az[tb:az.size - te]
        row0 = raz[-n - tb:raz.size - tb]
    else:
        col0 = az
        row0 = raz[-n:]
    return toeplitz(col0, row0)
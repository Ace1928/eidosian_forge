import warnings
import cupy
import numpy
from cupy.cuda import thrust
def sort_complex(a):
    """Sort a complex array using the real part first,
    then the imaginary part.

    Args:
        a (cupy.ndarray): Array to be sorted.

    Returns:
        cupy.ndarray: sorted complex array.

    .. seealso:: :func:`numpy.sort_complex`

    """
    if a.dtype.char in 'bhBHF':
        a = a.astype('F')
    else:
        a = a.astype('D')
    a.sort()
    return a
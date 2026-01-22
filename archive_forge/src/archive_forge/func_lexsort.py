import warnings
import cupy
import numpy
from cupy.cuda import thrust
def lexsort(keys):
    """Perform an indirect sort using an array of keys.

    Args:
        keys (cupy.ndarray): ``(k, N)`` array containing ``k`` ``(N,)``-shaped
            arrays. The ``k`` different "rows" to be sorted. The last row is
            the primary sort key.

    Returns:
        cupy.ndarray: Array of indices that sort the keys.

    .. note::
        For its implementation reason, ``cupy.lexsort`` currently supports only
        keys with their rank of one or two and does not support ``axis``
        parameter that ``numpy.lexsort`` supports.

    .. seealso:: :func:`numpy.lexsort`

    """
    if not cupy.cuda.thrust.available:
        raise RuntimeError('Thrust is needed to use cupy.lexsort. Please install CUDA Toolkit with Thrust then reinstall CuPy after uninstalling it.')
    if keys.ndim == ():
        raise TypeError('need sequence of keys with len > 0 in lexsort')
    if keys.ndim == 1:
        return cupy.array(0, dtype=numpy.intp)
    if keys.ndim > 2:
        raise NotImplementedError('Keys with the rank of three or more is not supported in lexsort')
    if not keys.flags.c_contiguous:
        keys = keys.copy('C')
    idx_array = cupy.ndarray(keys._shape[1:], dtype=numpy.intp)
    k = keys._shape[0]
    n = keys._shape[1]
    thrust.lexsort(keys.dtype, idx_array.data.ptr, keys.data.ptr, k, n)
    return idx_array
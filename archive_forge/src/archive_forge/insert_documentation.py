import numpy
import cupy
from cupy import _core

    Return the indices to access the main diagonal of an n-dimensional array.
    See `diag_indices` for full details.

    Args:
        arr (cupy.ndarray): At least 2-D.

    .. seealso:: :func:`numpy.diag_indices_from`

    
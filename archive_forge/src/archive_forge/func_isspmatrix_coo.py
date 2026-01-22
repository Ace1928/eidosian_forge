import numpy
import cupy
from cupy import _core
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def isspmatrix_coo(x):
    """Checks if a given matrix is of COO format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.coo_matrix`.

    """
    return isinstance(x, coo_matrix)
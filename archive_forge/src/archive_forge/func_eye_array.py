import numbers
import math
import numpy as np
from scipy._lib._util import check_random_state, rng_integers
from ._sputils import upcast, get_index_dtype, isscalarlike
from ._sparsetools import csr_hstack
from ._bsr import bsr_matrix, bsr_array
from ._coo import coo_matrix, coo_array
from ._csc import csc_matrix, csc_array
from ._csr import csr_matrix, csr_array
from ._dia import dia_matrix, dia_array
from ._base import issparse, sparray
def eye_array(m, n=None, *, k=0, dtype=float, format=None):
    """Identity matrix in sparse array format

    Return a sparse array with ones on diagonal.
    Specifically a sparse array (m x n) where the kth diagonal
    is all ones and everything else is zeros.

    Parameters
    ----------
    m : int or tuple of ints
        Number of rows requested.
    n : int, optional
        Number of columns. Default: `m`.
    k : int, optional
        Diagonal to place ones on. Default: 0 (main diagonal).
    dtype : dtype, optional
        Data type of the array
    format : str, optional (default: "dia")
        Sparse format of the result, e.g., format="csr", etc.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> sp.sparse.eye_array(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.eye_array(3, dtype=np.int8)
    <3x3 sparse array of type '<class 'numpy.int8'>'
            with 3 stored elements (1 diagonals) in DIAgonal format>

    """
    return _eye(m, n, k, dtype, format)
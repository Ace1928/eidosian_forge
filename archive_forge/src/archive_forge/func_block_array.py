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
def block_array(blocks, *, format=None, dtype=None):
    """
    Build a sparse array from sparse sub-blocks

    Parameters
    ----------
    blocks : array_like
        Grid of sparse arrays with compatible shapes.
        An entry of None implies an all-zero array.
    format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
        The sparse format of the result (e.g. "csr"). By default an
        appropriate sparse array format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output array. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    block : sparse array

    See Also
    --------
    block_diag : specify blocks along the main diagonals
    diags : specify (possibly offset) diagonals

    Examples
    --------
    >>> from scipy.sparse import coo_array, block_array
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> block_array([[A, B], [None, C]]).toarray()
    array([[1, 2, 5],
           [3, 4, 6],
           [0, 0, 7]])

    >>> block_array([[A, None], [None, C]]).toarray()
    array([[1, 2, 0],
           [3, 4, 0],
           [0, 0, 7]])

    """
    return _block(blocks, format, dtype)
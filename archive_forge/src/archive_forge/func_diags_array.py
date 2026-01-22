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
def diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=None):
    """
    Construct a sparse array from diagonals.

    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the array diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set:
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square array large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse array format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the array.

    Notes
    -----
    The result from `diags_array` is the sparse equivalent of::

        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])

    Repeated diagonal offsets are disallowed.

    .. versionadded:: 1.11

    Examples
    --------
    >>> from scipy.sparse import diags_array
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags_array(diagonals, offsets=[0, -1, 2]).toarray()
    array([[1, 0, 1, 0],
           [1, 2, 0, 2],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    Broadcasting of scalars is supported (but shape needs to be
    specified):

    >>> diags_array([1, -2, 1], offsets=[-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])


    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:

    >>> diags_array([1, 2, 3], offsets=1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])
    """
    if isscalarlike(offsets):
        if len(diagonals) == 0 or isscalarlike(diagonals[0]):
            diagonals = [np.atleast_1d(diagonals)]
        else:
            raise ValueError('Different number of diagonals and offsets.')
    else:
        diagonals = list(map(np.atleast_1d, diagonals))
    offsets = np.atleast_1d(offsets)
    if len(diagonals) != len(offsets):
        raise ValueError('Different number of diagonals and offsets.')
    if shape is None:
        m = len(diagonals[0]) + abs(int(offsets[0]))
        shape = (m, m)
    if dtype is None:
        dtype = np.common_type(*diagonals)
    m, n = shape
    M = max([min(m + offset, n - offset) + max(0, offset) for offset in offsets])
    M = max(0, M)
    data_arr = np.zeros((len(offsets), M), dtype=dtype)
    K = min(m, n)
    for j, diagonal in enumerate(diagonals):
        offset = offsets[j]
        k = max(0, offset)
        length = min(m + offset, n - offset, K)
        if length < 0:
            raise ValueError('Offset %d (index %d) out of bounds' % (offset, j))
        try:
            data_arr[j, k:k + length] = diagonal[..., :length]
        except ValueError as e:
            if len(diagonal) != length and len(diagonal) != 1:
                raise ValueError('Diagonal length (index %d: %d at offset %d) does not agree with array size (%d, %d).' % (j, len(diagonal), offset, m, n)) from e
            raise
    return dia_array((data_arr, offsets), shape=(m, n)).asformat(format)
import numpy
import cupy
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _dia
from cupyx.scipy.sparse import _sputils
def kronsum(A, B, format=None):
    """Kronecker sum of sparse matrices A and B.

    Kronecker sum is the sum of two Kronecker products
    ``kron(I_n, A) + kron(B, I_m)``, where ``I_n`` and ``I_m`` are identity
    matrices.

    Args:
        A (cupyx.scipy.sparse.spmatrix): a sparse matrix.
        B (cupyx.scipy.sparse.spmatrix): a sparse matrix.
        format (str): the format of the returned sparse matrix.

    Returns:
        cupyx.scipy.sparse.spmatrix:
            Generated sparse matrix with the specified ``format``.

    .. seealso:: :func:`scipy.sparse.kronsum`

    """
    A = _coo.coo_matrix(A)
    B = _coo.coo_matrix(B)
    if A.shape[0] != A.shape[1]:
        raise ValueError('A is not square matrix')
    if B.shape[0] != B.shape[1]:
        raise ValueError('B is not square matrix')
    dtype = _sputils.upcast(A.dtype, B.dtype)
    L = kron(eye(B.shape[0], dtype=dtype), A, format=format)
    R = kron(B, eye(A.shape[0], dtype=dtype), format=format)
    return (L + R).asformat(format)
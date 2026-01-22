import numpy
import cupy
import cupyx.cusolver
from cupy import cublas
from cupyx import cusparse
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import _interface
from cupyx.scipy.sparse.linalg._iterative import _make_system
import warnings
def spsolve(A, b):
    """Solves a sparse linear system ``A x = b``

    Args:
        A (cupyx.scipy.sparse.spmatrix):
            Sparse matrix with dimension ``(M, M)``.
        b (cupy.ndarray):
            Dense vector or matrix with dimension ``(M)`` or ``(M, N)``.

    Returns:
        cupy.ndarray:
            Solution to the system ``A x = b``.
    """
    if not cupyx.cusolver.check_availability('csrlsvqr'):
        raise NotImplementedError
    if not sparse.isspmatrix(A):
        raise TypeError('A must be cupyx.scipy.sparse.spmatrix')
    if not isinstance(b, cupy.ndarray):
        raise TypeError('b must be cupy.ndarray')
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix (A.shape: {})'.format(A.shape))
    if not (b.ndim == 1 or b.ndim == 2):
        raise ValueError('Invalid b.shape (b.shape: {})'.format(b.shape))
    if A.shape[0] != b.shape[0]:
        raise ValueError('matrix dimension mismatch (A.shape: {}, b.shape: {})'.format(A.shape, b.shape))
    if not sparse.isspmatrix_csr(A):
        warnings.warn('CSR format is required. Converting to CSR format.', sparse.SparseEfficiencyWarning)
        A = A.tocsr()
    A.sum_duplicates()
    b = b.astype(A.dtype, copy=False)
    if b.ndim > 1:
        res = cupy.empty_like(b)
        for j in range(res.shape[1]):
            res[:, j] = cupyx.cusolver.csrlsvqr(A, b[:, j])
        res = cupy.asarray(res, order='F')
        return res
    else:
        return cupyx.cusolver.csrlsvqr(A, b)
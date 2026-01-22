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
def spsolve_triangular(A, b, lower=True, overwrite_A=False, overwrite_b=False, unit_diagonal=False):
    """Solves a sparse triangular system ``A x = b``.

    Args:
        A (cupyx.scipy.sparse.spmatrix):
            Sparse matrix with dimension ``(M, M)``.
        b (cupy.ndarray):
            Dense vector or matrix with dimension ``(M)`` or ``(M, K)``.
        lower (bool):
            Whether ``A`` is a lower or upper trinagular matrix.
            If True, it is lower triangular, otherwise, upper triangular.
        overwrite_A (bool):
            (not supported)
        overwrite_b (bool):
            Allows overwriting data in ``b``.
        unit_diagonal (bool):
            If True, diagonal elements of ``A`` are assumed to be 1 and will
            not be referenced.

    Returns:
        cupy.ndarray:
            Solution to the system ``A x = b``. The shape is the same as ``b``.
    """
    if not (cusparse.check_availability('spsm') or cusparse.check_availability('csrsm2')):
        raise NotImplementedError
    if not sparse.isspmatrix(A):
        raise TypeError('A must be cupyx.scipy.sparse.spmatrix')
    if not isinstance(b, cupy.ndarray):
        raise TypeError('b must be cupy.ndarray')
    if A.shape[0] != A.shape[1]:
        raise ValueError(f'A must be a square matrix (A.shape: {A.shape})')
    if b.ndim not in [1, 2]:
        raise ValueError(f'b must be 1D or 2D array (b.shape: {b.shape})')
    if A.shape[0] != b.shape[0]:
        raise ValueError(f'The size of dimensions of A must be equal to the size of the first dimension of b (A.shape: {A.shape}, b.shape: {b.shape})')
    if A.dtype.char not in 'fdFD':
        raise TypeError(f'unsupported dtype (actual: {A.dtype})')
    if cusparse.check_availability('spsm') and _should_use_spsm(b):
        if not (sparse.isspmatrix_csr(A) or sparse.isspmatrix_csc(A) or sparse.isspmatrix_coo(A)):
            warnings.warn('CSR, CSC or COO format is required. Converting to CSR format.', sparse.SparseEfficiencyWarning)
            A = A.tocsr()
        A.sum_duplicates()
        x = cusparse.spsm(A, b, lower=lower, unit_diag=unit_diagonal)
    elif cusparse.check_availability('csrsm2'):
        if not (sparse.isspmatrix_csr(A) or sparse.isspmatrix_csc(A)):
            warnings.warn('CSR or CSC format is required. Converting to CSR format.', sparse.SparseEfficiencyWarning)
            A = A.tocsr()
        A.sum_duplicates()
        if overwrite_b and A.dtype == b.dtype and (b._c_contiguous or b._f_contiguous):
            x = b
        else:
            x = b.astype(A.dtype, copy=True)
        cusparse.csrsm2(A, x, lower=lower, unit_diag=unit_diagonal)
    else:
        assert False
    if x.dtype.char in 'fF':
        dtype = numpy.promote_types(x.dtype, 'float64')
        x = x.astype(dtype)
    return x
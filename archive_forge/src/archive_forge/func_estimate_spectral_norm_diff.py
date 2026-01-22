import scipy.linalg._interpolative_backend as _backend
import numpy as np
import sys
def estimate_spectral_norm_diff(A, B, its=20):
    """
    Estimate spectral norm of the difference of two matrices by the randomized
    power method.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_diffsnorm` and
        :func:`_backend.idz_diffsnorm`.

    Parameters
    ----------
    A : :class:`scipy.sparse.linalg.LinearOperator`
        First matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
        `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    B : :class:`scipy.sparse.linalg.LinearOperator`
        Second matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with
        the `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    its : int, optional
        Number of power method iterations.

    Returns
    -------
    float
        Spectral norm estimate of matrix difference.
    """
    from scipy.sparse.linalg import aslinearoperator
    A = aslinearoperator(A)
    B = aslinearoperator(B)
    m, n = A.shape

    def matvec1(x):
        return A.matvec(x)

    def matveca1(x):
        return A.rmatvec(x)

    def matvec2(x):
        return B.matvec(x)

    def matveca2(x):
        return B.rmatvec(x)
    if _is_real(A):
        return _backend.idd_diffsnorm(m, n, matveca1, matveca2, matvec1, matvec2, its=its)
    else:
        return _backend.idz_diffsnorm(m, n, matveca1, matveca2, matvec1, matvec2, its=its)
import numpy as np
from scipy.linalg import (norm, get_lapack_funcs, solve_triangular,
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
def singular_leading_submatrix(A, U, k):
    """
    Compute term that makes the leading ``k`` by ``k``
    submatrix from ``A`` singular.

    Parameters
    ----------
    A : ndarray
        Symmetric matrix that is not positive definite.
    U : ndarray
        Upper triangular matrix resulting of an incomplete
        Cholesky decomposition of matrix ``A``.
    k : int
        Positive integer such that the leading k by k submatrix from
        `A` is the first non-positive definite leading submatrix.

    Returns
    -------
    delta : float
        Amount that should be added to the element (k, k) of the
        leading k by k submatrix of ``A`` to make it singular.
    v : ndarray
        A vector such that ``v.T B v = 0``. Where B is the matrix A after
        ``delta`` is added to its element (k, k).
    """
    delta = np.sum(U[:k - 1, k - 1] ** 2) - A[k - 1, k - 1]
    n = len(A)
    v = np.zeros(n)
    v[k - 1] = 1
    if k != 1:
        v[:k - 1] = solve_triangular(U[:k - 1, :k - 1], -U[:k - 1, k - 1])
    return (delta, v)
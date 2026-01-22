import scipy.linalg._interpolative_backend as _backend
import numpy as np
import sys
def estimate_rank(A, eps):
    """
    Estimate matrix rank to a specified relative precision using randomized
    methods.

    The matrix `A` can be given as either a :class:`numpy.ndarray` or a
    :class:`scipy.sparse.linalg.LinearOperator`, with different algorithms used
    for each case. If `A` is of type :class:`numpy.ndarray`, then the output
    rank is typically about 8 higher than the actual numerical rank.

    ..  This function automatically detects the form of the input parameters and
        passes them to the appropriate backend. For details,
        see :func:`_backend.idd_estrank`, :func:`_backend.idd_findrank`,
        :func:`_backend.idz_estrank`, and :func:`_backend.idz_findrank`.

    Parameters
    ----------
    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
        Matrix whose rank is to be estimated, given as either a
        :class:`numpy.ndarray` or a :class:`scipy.sparse.linalg.LinearOperator`
        with the `rmatvec` method (to apply the matrix adjoint).
    eps : float
        Relative error for numerical rank definition.

    Returns
    -------
    int
        Estimated matrix rank.
    """
    from scipy.sparse.linalg import LinearOperator
    real = _is_real(A)
    if isinstance(A, np.ndarray):
        if real:
            rank = _backend.idd_estrank(eps, A)
        else:
            rank = _backend.idz_estrank(eps, A)
        if rank == 0:
            rank = min(A.shape)
        return rank
    elif isinstance(A, LinearOperator):
        m, n = A.shape
        matveca = A.rmatvec
        if real:
            return _backend.idd_findrank(eps, m, n, matveca)
        else:
            return _backend.idz_findrank(eps, m, n, matveca)
    else:
        raise _TYPE_ERROR
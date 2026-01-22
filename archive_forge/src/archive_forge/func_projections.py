from scipy.sparse import (bmat, csc_matrix, eye, issparse)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
from warnings import warn
def projections(A, method=None, orth_tol=1e-12, max_refin=3, tol=1e-15):
    """Return three linear operators related with a given matrix A.

    Parameters
    ----------
    A : sparse matrix (or ndarray), shape (m, n)
        Matrix ``A`` used in the projection.
    method : string, optional
        Method used for compute the given linear
        operators. Should be one of:

            - 'NormalEquation': The operators
               will be computed using the
               so-called normal equation approach
               explained in [1]_. In order to do
               so the Cholesky factorization of
               ``(A A.T)`` is computed. Exclusive
               for sparse matrices.
            - 'AugmentedSystem': The operators
               will be computed using the
               so-called augmented system approach
               explained in [1]_. Exclusive
               for sparse matrices.
            - 'QRFactorization': Compute projections
               using QR factorization. Exclusive for
               dense matrices.
            - 'SVDFactorization': Compute projections
               using SVD factorization. Exclusive for
               dense matrices.

    orth_tol : float, optional
        Tolerance for iterative refinements.
    max_refin : int, optional
        Maximum number of iterative refinements.
    tol : float, optional
        Tolerance for singular values.

    Returns
    -------
    Z : LinearOperator, shape (n, n)
        Null-space operator. For a given vector ``x``,
        the null space operator is equivalent to apply
        a projection matrix ``P = I - A.T inv(A A.T) A``
        to the vector. It can be shown that this is
        equivalent to project ``x`` into the null space
        of A.
    LS : LinearOperator, shape (m, n)
        Least-squares operator. For a given vector ``x``,
        the least-squares operator is equivalent to apply a
        pseudoinverse matrix ``pinv(A.T) = inv(A A.T) A``
        to the vector. It can be shown that this vector
        ``pinv(A.T) x`` is the least_square solution to
        ``A.T y = x``.
    Y : LinearOperator, shape (n, m)
        Row-space operator. For a given vector ``x``,
        the row-space operator is equivalent to apply a
        projection matrix ``Q = A.T inv(A A.T)``
        to the vector.  It can be shown that this
        vector ``y = Q x``  the minimum norm solution
        of ``A y = x``.

    Notes
    -----
    Uses iterative refinements described in [1]
    during the computation of ``Z`` in order to
    cope with the possibility of large roundoff errors.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
        "On the solution of equality constrained quadratic
        programming problems arising in optimization."
        SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """
    m, n = np.shape(A)
    if m * n == 0:
        A = csc_matrix(A)
    if issparse(A):
        if method is None:
            method = 'AugmentedSystem'
        if method not in ('NormalEquation', 'AugmentedSystem'):
            raise ValueError('Method not allowed for sparse matrix.')
        if method == 'NormalEquation' and (not sksparse_available):
            warnings.warn("Only accepts 'NormalEquation' option when scikit-sparse is available. Using 'AugmentedSystem' option instead.", ImportWarning, stacklevel=3)
            method = 'AugmentedSystem'
    else:
        if method is None:
            method = 'QRFactorization'
        if method not in ('QRFactorization', 'SVDFactorization'):
            raise ValueError('Method not allowed for dense array.')
    if method == 'NormalEquation':
        null_space, least_squares, row_space = normal_equation_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == 'AugmentedSystem':
        null_space, least_squares, row_space = augmented_system_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == 'QRFactorization':
        null_space, least_squares, row_space = qr_factorization_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == 'SVDFactorization':
        null_space, least_squares, row_space = svd_factorization_projections(A, m, n, orth_tol, max_refin, tol)
    Z = LinearOperator((n, n), null_space)
    LS = LinearOperator((m, n), least_squares)
    Y = LinearOperator((n, m), row_space)
    return (Z, LS, Y)
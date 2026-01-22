from scipy.sparse import (bmat, csc_matrix, eye, issparse)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
from warnings import warn
def augmented_system_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A - ``AugmentedSystem``."""
    K = csc_matrix(bmat([[eye(n), A.T], [A, None]]))
    try:
        solve = scipy.sparse.linalg.factorized(K)
    except RuntimeError:
        warn('Singular Jacobian matrix. Using dense SVD decomposition to perform the factorizations.', stacklevel=3)
        return svd_factorization_projections(A.toarray(), m, n, orth_tol, max_refin, tol)

    def null_space(x):
        v = np.hstack([x, np.zeros(m)])
        lu_sol = solve(v)
        z = lu_sol[:n]
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            new_v = v - K.dot(lu_sol)
            lu_update = solve(new_v)
            lu_sol += lu_update
            z = lu_sol[:n]
            k += 1
        return z

    def least_squares(x):
        v = np.hstack([x, np.zeros(m)])
        lu_sol = solve(v)
        return lu_sol[n:m + n]

    def row_space(x):
        v = np.hstack([np.zeros(n), x])
        lu_sol = solve(v)
        return lu_sol[:n]
    return (null_space, least_squares, row_space)
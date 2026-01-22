from scipy.sparse import (bmat, csc_matrix, eye, issparse)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
from warnings import warn
def qr_factorization_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A using ``QRFactorization`` approach.
    """
    Q, R, P = scipy.linalg.qr(A.T, pivoting=True, mode='economic')
    if np.linalg.norm(R[-1, :], np.inf) < tol:
        warn('Singular Jacobian matrix. Using SVD decomposition to ' + 'perform the factorizations.', stacklevel=3)
        return svd_factorization_projections(A, m, n, orth_tol, max_refin, tol)

    def null_space(x):
        aux1 = Q.T.dot(x)
        aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
        v = np.zeros(m)
        v[P] = aux2
        z = x - A.T.dot(v)
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            aux1 = Q.T.dot(z)
            aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
            v[P] = aux2
            z = z - A.T.dot(v)
            k += 1
        return z

    def least_squares(x):
        aux1 = Q.T.dot(x)
        aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
        z = np.zeros(m)
        z[P] = aux2
        return z

    def row_space(x):
        aux1 = x[P]
        aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False, trans='T')
        z = Q.dot(aux2)
        return z
    return (null_space, least_squares, row_space)
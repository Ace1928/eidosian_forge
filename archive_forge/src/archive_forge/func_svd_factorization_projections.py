from scipy.sparse import (bmat, csc_matrix, eye, issparse)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
from warnings import warn
def svd_factorization_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A using ``SVDFactorization`` approach.
    """
    U, s, Vt = scipy.linalg.svd(A, full_matrices=False)
    U = U[:, s > tol]
    Vt = Vt[s > tol, :]
    s = s[s > tol]

    def null_space(x):
        aux1 = Vt.dot(x)
        aux2 = 1 / s * aux1
        v = U.dot(aux2)
        z = x - A.T.dot(v)
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            aux1 = Vt.dot(z)
            aux2 = 1 / s * aux1
            v = U.dot(aux2)
            z = z - A.T.dot(v)
            k += 1
        return z

    def least_squares(x):
        aux1 = Vt.dot(x)
        aux2 = 1 / s * aux1
        z = U.dot(aux2)
        return z

    def row_space(x):
        aux1 = U.T.dot(x)
        aux2 = 1 / s * aux1
        z = Vt.T.dot(aux2)
        return z
    return (null_space, least_squares, row_space)
from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def phi_and_derivative(alpha, suf, s, Delta):
    """Function of which to find zero.

        It is defined as "norm of regularized (by alpha) least-squares
        solution minus `Delta`". Refer to [1]_.
        """
    denom = s ** 2 + alpha
    p_norm = norm(suf / denom)
    phi = p_norm - Delta
    phi_prime = -np.sum(suf ** 2 / denom ** 3) / p_norm
    return (phi, phi_prime)
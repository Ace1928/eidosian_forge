from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def right_multiply(J, d, copy=True):
    """Compute J diag(d).

    If `copy` is False, `J` is modified in place (unless being LinearOperator).
    """
    if copy and (not isinstance(J, LinearOperator)):
        J = J.copy()
    if issparse(J):
        J.data *= d.take(J.indices, mode='clip')
    elif isinstance(J, LinearOperator):
        J = right_multiplied_operator(J, d)
    else:
        J *= d
    return J
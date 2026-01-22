from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def scale_for_robust_loss_function(J, f, rho):
    """Scale Jacobian and residuals for a robust loss function.

    Arrays are modified in place.
    """
    J_scale = rho[1] + 2 * rho[2] * f ** 2
    J_scale[J_scale < EPS] = EPS
    J_scale **= 0.5
    f *= rho[1] / J_scale
    return (left_multiply(J, J_scale, copy=False), f)
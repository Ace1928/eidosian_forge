import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def gradient_and_jacobian(self, z):
    """Returns scaled gradient.

        Return scaled gradient:
            gradient = [             grad(x)             ]
                       [ -barrier_parameter*ones(n_ineq) ]
        and scaled Jacobian matrix:
            jacobian = [  jac_eq(x)  0  ]
                       [ jac_ineq(x) S  ]
        Both of them scaled by the previously defined scaling factor.
        """
    x = self.get_variables(z)
    s = self.get_slack(z)
    g = self.grad(x)
    J_eq, J_ineq = self.jac(x)
    return (self._compute_gradient(g), self._compute_jacobian(J_eq, J_ineq, s))
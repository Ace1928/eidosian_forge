import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def lagrangian_hessian_x(self, z, v):
    """Returns Lagrangian Hessian (in relation to `x`) -> Hx"""
    x = self.get_variables(z)
    v_eq = v[:self.n_eq]
    v_ineq = v[self.n_eq:self.n_eq + self.n_ineq]
    lagr_hess = self.lagr_hess
    return lagr_hess(x, v_eq, v_ineq)
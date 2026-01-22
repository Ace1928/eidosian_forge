import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def lagrangian_hessian_s(self, z, v):
    """Returns scaled Lagrangian Hessian (in relation to`s`) -> S Hs S"""
    s = self.get_slack(z)
    primal = self.barrier_parameter
    primal_dual = v[-self.n_ineq:] * s
    return np.where(v[-self.n_ineq:] > 0, primal_dual, primal)
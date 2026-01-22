import numpy as np
import scipy.sparse as sps
from ._numdiff import approx_derivative, group_columns
from ._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import atleast_nd, array_namespace
def update_hess():
    self._update_jac()
    if self.x_prev is not None and self.J_prev is not None:
        delta_x = self.x - self.x_prev
        delta_g = self.J.T.dot(self.v) - self.J_prev.T.dot(self.v)
        self.H.update(delta_x, delta_g)
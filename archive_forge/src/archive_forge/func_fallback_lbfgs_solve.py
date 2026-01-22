import warnings
from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg
import scipy.optimize
from ..._loss.loss import HalfSquaredError
from ...exceptions import ConvergenceWarning
from ...utils.optimize import _check_optimize_result
from .._linear_loss import LinearModelLoss
def fallback_lbfgs_solve(self, X, y, sample_weight):
    """Fallback solver in case of emergency.

        If a solver detects convergence problems, it may fall back to this methods in
        the hope to exit with success instead of raising an error.

        Sets:
            - self.coef
            - self.converged
        """
    opt_res = scipy.optimize.minimize(self.linear_loss.loss_gradient, self.coef, method='L-BFGS-B', jac=True, options={'maxiter': self.max_iter, 'maxls': 50, 'iprint': self.verbose - 1, 'gtol': self.tol, 'ftol': 64 * np.finfo(np.float64).eps}, args=(X, y, sample_weight, self.l2_reg_strength, self.n_threads))
    self.n_iter_ = _check_optimize_result('lbfgs', opt_res)
    self.coef = opt_res.x
    self.converged = opt_res.status == 0
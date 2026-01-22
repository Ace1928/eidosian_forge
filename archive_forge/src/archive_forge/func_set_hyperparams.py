import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ZeroPrior
def set_hyperparams(self, params):
    """Set hyperparameters of the regression.
        This is a list containing the parameters of the
        kernel and the regularization (noise)
        of the method as the last entry.
        """
    self.hyperparams = params
    self.kernel.set_params(params[:-1])
    self.noise = params[-1]
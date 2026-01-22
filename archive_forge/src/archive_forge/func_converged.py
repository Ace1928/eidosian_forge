import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def converged(self):
    """
        Return True if the solver has converged.
        """
    if np.any(np.isinf(self.population_energies)):
        return False
    return np.std(self.population_energies) <= self.atol + self.tol * np.abs(np.mean(self.population_energies))
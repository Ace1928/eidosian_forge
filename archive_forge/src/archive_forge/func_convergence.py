import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
@property
def convergence(self):
    """
        The standard deviation of the population energies divided by their
        mean.
        """
    if np.any(np.isinf(self.population_energies)):
        return np.inf
    return np.std(self.population_energies) / (np.abs(np.mean(self.population_energies)) + _MACHEPS)
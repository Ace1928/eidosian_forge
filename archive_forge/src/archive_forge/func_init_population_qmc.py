import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def init_population_qmc(self, qmc_engine):
    """Initializes the population with a QMC method.

        QMC methods ensures that each parameter is uniformly
        sampled over its range.

        Parameters
        ----------
        qmc_engine : str
            The QMC method to use for initialization. Can be one of
            ``latinhypercube``, ``sobol`` or ``halton``.

        """
    from scipy.stats import qmc
    rng = self.random_number_generator
    if qmc_engine == 'latinhypercube':
        sampler = qmc.LatinHypercube(d=self.parameter_count, seed=rng)
    elif qmc_engine == 'sobol':
        sampler = qmc.Sobol(d=self.parameter_count, seed=rng)
    elif qmc_engine == 'halton':
        sampler = qmc.Halton(d=self.parameter_count, seed=rng)
    else:
        raise ValueError(self.__init_error_msg)
    self.population = sampler.random(n=self.num_population_members)
    self.population_energies = np.full(self.num_population_members, np.inf)
    self._nfev = 0
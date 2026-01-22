import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def init_population_lhs(self):
    """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
    rng = self.random_number_generator
    segsize = 1.0 / self.num_population_members
    samples = segsize * rng.uniform(size=self.population_shape) + np.linspace(0.0, 1.0, self.num_population_members, endpoint=False)[:, np.newaxis]
    self.population = np.zeros_like(samples)
    for j in range(self.parameter_count):
        order = rng.permutation(range(self.num_population_members))
        self.population[:, j] = samples[order, j]
    self.population_energies = np.full(self.num_population_members, np.inf)
    self._nfev = 0
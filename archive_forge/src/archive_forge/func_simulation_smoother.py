import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
def simulation_smoother(self, simulation_output=None, method='kfs', results_class=None, prefix=None, nobs=-1, random_state=None, **kwargs):
    """
        Retrieve a simulation smoother for the statespace model.

        Parameters
        ----------
        simulation_output : int, optional
            Determines which simulation smoother output is calculated.
            Default is all (including state and disturbances).
        method : {'kfs', 'cfa'}, optional
            Method for simulation smoothing. If `method='kfs'`, then the
            simulation smoother is based on Kalman filtering and smoothing
            recursions. If `method='cfa'`, then the simulation smoother is
            based on the Cholesky Factor Algorithm (CFA) approach. The CFA
            approach is not applicable to all state space models, but can be
            faster for the cases in which it is supported.
        results_class : class, optional
            Default results class to use to save output of simulation
            smoothing. Default is `SimulationSmoothResults`. If specified,
            class must extend from `SimulationSmoothResults`.
        prefix : str
            The prefix of the datatype. Usually only used internally.
        nobs : int
            The number of observations to simulate. If set to anything other
            than -1, only simulation will be performed (i.e. simulation
            smoothing will not be performed), so that only the `generated_obs`
            and `generated_state` attributes will be available.
        random_state : {None, int, Generator, RandomState}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``numpy.random.RandomState`` instance
            is used, seeded with `seed`.
            If `seed` is already a ``numpy.random.Generator`` or
            ``numpy.random.RandomState`` instance then that instance is used.
        **kwargs
            Additional keyword arguments, used to set the simulation output.
            See `set_simulation_output` for more details.

        Returns
        -------
        SimulationSmoothResults
        """
    method = method.lower()
    if method == 'cfa':
        if simulation_output not in [None, 1, -1]:
            raise ValueError('Can only retrieve simulations of the state vector using the CFA simulation smoother.')
        return CFASimulationSmoother(self)
    elif method != 'kfs':
        raise ValueError('Invalid simulation smoother method "%s". Valid methods are "kfs" or "cfa".' % method)
    if results_class is None:
        results_class = self.simulation_smooth_results_class
    if not issubclass(results_class, SimulationSmoothResults):
        raise ValueError('Invalid results class provided.')
    prefix, dtype, create_smoother, create_filter, create_statespace = self._initialize_smoother()
    simulation_output = self.get_simulation_output(simulation_output, **kwargs)
    smoother_output = kwargs.get('smoother_output', simulation_output)
    filter_method = kwargs.get('filter_method', self.filter_method)
    inversion_method = kwargs.get('inversion_method', self.inversion_method)
    stability_method = kwargs.get('stability_method', self.stability_method)
    conserve_memory = kwargs.get('conserve_memory', self.conserve_memory)
    filter_timing = kwargs.get('filter_timing', self.filter_timing)
    loglikelihood_burn = kwargs.get('loglikelihood_burn', self.loglikelihood_burn)
    tolerance = kwargs.get('tolerance', self.tolerance)
    cls = self.prefix_simulation_smoother_map[prefix]
    simulation_smoother = cls(self._statespaces[prefix], filter_method, inversion_method, stability_method, conserve_memory, filter_timing, tolerance, loglikelihood_burn, smoother_output, simulation_output, nobs)
    results = results_class(self, simulation_smoother, random_state=random_state)
    return results
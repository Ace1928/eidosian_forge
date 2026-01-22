import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
@property
def generated_obs(self):
    """
        Generated vector of observations by iterating on the observation and
        transition equations, given a random initial state draw and random
        disturbance draws.

        Notes
        -----

        .. math::

            y_t^+ = d_t + Z_t \\alpha_t^+ + \\varepsilon_t^+
        """
    if self._generated_obs is None:
        self._generated_obs = np.array(self._simulation_smoother.generated_obs, copy=True)
    return self._generated_obs
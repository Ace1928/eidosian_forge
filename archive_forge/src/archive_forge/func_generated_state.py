import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
@property
def generated_state(self):
    """
        Generated vector of states by iterating on the transition equation,
        given a random initial state draw and random disturbance draws.

        Notes
        -----

        .. math::

            \\alpha_{t+1}^+ = c_t + T_t \\alpha_t^+ + \\eta_t^+
        """
    if self._generated_state is None:
        self._generated_state = np.array(self._simulation_smoother.generated_state, copy=True)
    return self._generated_state
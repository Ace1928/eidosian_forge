import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
@property
def simulated_state(self):
    """
        Random draw of the state vector from its conditional distribution.

        Notes
        -----

        .. math::

            \\alpha ~ p(\\alpha \\mid Y_n)
        """
    if self._simulated_state is None:
        self._simulated_state = np.array(self._simulation_smoother.simulated_state, copy=True)
    return self._simulated_state
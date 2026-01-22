import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
@property
def simulated_measurement_disturbance(self):
    """
        Random draw of the measurement disturbance vector from its conditional
        distribution.

        Notes
        -----

        .. math::

            \\varepsilon ~ N(\\hat \\varepsilon, Var(\\hat \\varepsilon \\mid Y_n))
        """
    if self._simulated_measurement_disturbance is None:
        self._simulated_measurement_disturbance = np.array(self._simulation_smoother.simulated_measurement_disturbance, copy=True)
    return self._simulated_measurement_disturbance
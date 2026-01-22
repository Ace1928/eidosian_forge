import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
@property
def generated_measurement_disturbance(self):
    """
        Randomly drawn measurement disturbance variates

        Used to construct `generated_obs`.

        Notes
        -----

        .. math::

           \\varepsilon_t^+ ~ N(0, H_t)

        If `disturbance_variates` were provided to the `simulate()` method,
        then this returns those variates (which were N(0,1)) transformed to the
        distribution above.
        """
    if self._generated_measurement_disturbance is None:
        self._generated_measurement_disturbance = np.array(self._simulation_smoother.measurement_disturbance_variates, copy=True).reshape(self.model.nobs, self.model.k_endog)
    return self._generated_measurement_disturbance
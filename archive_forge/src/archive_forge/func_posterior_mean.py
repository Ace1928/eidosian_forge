import numpy as np
from . import tools
@property
def posterior_mean(self):
    """
        Posterior mean of the states conditional on the data

        Notes
        -----

        .. math::

            \\hat \\alpha_t = E[\\alpha_t \\mid Y^n ]

        This posterior mean is identical to the `smoothed_state` computed by
        the Kalman smoother.
        """
    if self._posterior_mean is None:
        self._posterior_mean = np.array(self._simulation_smoother.posterior_mean, copy=True)
    return self._posterior_mean
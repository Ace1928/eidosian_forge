import warnings
from scipy import linalg, special
from scipy._lib._util import check_random_state
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
import numpy as np
from . import _mvn
from ._stats import gaussian_kernel_estimate, gaussian_kernel_estimate_log
from scipy.special import logsumexp  # noqa: F401
def silverman_factor(self):
    """Compute the Silverman factor.

        Returns
        -------
        s : float
            The silverman factor.
        """
    return power(self.neff * (self.d + 2.0) / 4.0, -1.0 / (self.d + 4))
import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state
def cdfcond_2g1(self, u, args=()):
    """Conditional cdf of second component given the value of first.
        """
    u = self._handle_u(u)
    th, = self._handle_args(args)
    if u.shape[-1] == 2:
        u1, u2 = (u[..., 0], u[..., 1])
        cdfc = np.exp(-th * u1)
        cdfc /= np.expm1(-th) / np.expm1(-th * u2) + np.expm1(-th * u1)
        return cdfc
    else:
        raise NotImplementedError('u needs to be bivariate (2 columns)')
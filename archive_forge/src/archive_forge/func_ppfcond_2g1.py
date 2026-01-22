import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state
def ppfcond_2g1(self, q, u1, args=()):
    """Conditional pdf of second component given the value of first.
        """
    u1 = np.asarray(u1)
    th, = self._handle_args(args)
    if u1.shape[-1] == 1:
        ppfc = -np.log(1 + np.expm1(-th) / ((1 / q - 1) * np.exp(-th * u1) + 1)) / th
        return ppfc
    else:
        raise NotImplementedError('u needs to be bivariate (2 columns)')
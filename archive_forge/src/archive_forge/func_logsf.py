from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
def logsf(self, k, *args, **kwds):
    """Log of the survival function of the given RV.

        Returns the log of the "survival function," defined as 1 - `cdf`,
        evaluated at `k`.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logsf : ndarray
            Log of the survival function evaluated at `k`.

        """
    args, loc, _ = self._parse_args(*args, **kwds)
    k, loc = map(asarray, (k, loc))
    args = tuple(map(asarray, args))
    _a, _b = self._get_support(*args)
    k = asarray(k - loc)
    cond0 = self._argcheck(*args)
    cond1 = (k >= _a) & (k < _b)
    cond2 = (k < _a) & cond0
    cond = cond0 & cond1
    output = empty(shape(cond), 'd')
    output.fill(-inf)
    place(output, 1 - cond0 + np.isnan(k), self.badvalue)
    place(output, cond2, 0.0)
    if np.any(cond):
        goodargs = argsreduce(cond, *(k,) + args)
        place(output, cond, self._logsf(*goodargs))
    if output.ndim == 0:
        return output[()]
    return output
import operator
import numpy as np
import math
import warnings
from collections import defaultdict
from heapq import heapify, heappop
from numpy import (pi, asarray, floor, isscalar, iscomplex, real,
from . import _ufuncs
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
from . import _specfun
from ._comb import _comb_int
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def factorial2(n, exact=False):
    """Double factorial.

    This is the factorial with every second value skipped.  E.g., ``7!! = 7 * 5
    * 3 * 1``.  It can be approximated numerically as::

      n!! = 2 ** (n / 2) * gamma(n / 2 + 1) * sqrt(2 / pi)  n odd
          = 2 ** (n / 2) * gamma(n / 2 + 1)                 n even
          = 2 ** (n / 2) * (n / 2)!                         n even

    Parameters
    ----------
    n : int or array_like
        Calculate ``n!!``.  If ``n < 0``, the return value is 0.
    exact : bool, optional
        The result can be approximated rapidly using the gamma-formula
        above (default).  If `exact` is set to True, calculate the
        answer exactly using integer arithmetic.

    Returns
    -------
    nff : float or int
        Double factorial of `n`, as an int or a float depending on
        `exact`.

    Examples
    --------
    >>> from scipy.special import factorial2
    >>> factorial2(7, exact=False)
    array(105.00000000000001)
    >>> factorial2(7, exact=True)
    105

    """

    def _approx(n):
        val = np.power(2, n / 2) * gamma(n / 2 + 1)
        mask = np.ones_like(n, dtype=np.float64)
        mask[n % 2 == 1] = sqrt(2 / pi)
        return val * mask
    if np.ndim(n) == 0 and (not isinstance(n, np.ndarray)):
        if n is None or np.isnan(n):
            return np.nan
        elif not np.issubdtype(type(n), np.integer):
            msg = 'factorial2 does not support non-integral scalar arguments'
            raise ValueError(msg)
        elif n < 0:
            return 0
        elif n in {0, 1}:
            return 1
        if exact:
            return _range_prod(1, n, k=2)
        return _approx(n)
    n = asarray(n)
    if n.size == 0:
        return n
    if not np.issubdtype(n.dtype, np.integer):
        raise ValueError('factorial2 does not support non-integral arrays')
    if exact:
        return _exact_factorialx_array(n, k=2)
    vals = zeros(n.shape)
    cond = n >= 0
    n_to_compute = extract(cond, n)
    place(vals, cond, _approx(n_to_compute))
    return vals
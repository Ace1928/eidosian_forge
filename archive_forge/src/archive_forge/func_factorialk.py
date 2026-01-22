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
def factorialk(n, k, exact=True):
    """Multifactorial of n of order k, n(!!...!).

    This is the multifactorial of n skipping k values.  For example,

      factorialk(17, 4) = 17!!!! = 17 * 13 * 9 * 5 * 1

    In particular, for any integer ``n``, we have

      factorialk(n, 1) = factorial(n)

      factorialk(n, 2) = factorial2(n)

    Parameters
    ----------
    n : int or array_like
        Calculate multifactorial. If `n` < 0, the return value is 0.
    k : int
        Order of multifactorial.
    exact : bool, optional
        If exact is set to True, calculate the answer exactly using
        integer arithmetic.

    Returns
    -------
    val : int
        Multifactorial of `n`.

    Raises
    ------
    NotImplementedError
        Raises when exact is False

    Examples
    --------
    >>> from scipy.special import factorialk
    >>> factorialk(5, 1, exact=True)
    120
    >>> factorialk(5, 3, exact=True)
    10

    """
    if not np.issubdtype(type(k), np.integer) or k < 1:
        raise ValueError(f'k must be a positive integer, received: {k}')
    if not exact:
        raise NotImplementedError
    helpmsg = ''
    if k in {1, 2}:
        func = 'factorial' if k == 1 else 'factorial2'
        helpmsg = f'\nYou can try to use {func} instead'
    if np.ndim(n) == 0 and (not isinstance(n, np.ndarray)):
        if n is None or np.isnan(n):
            return np.nan
        elif not np.issubdtype(type(n), np.integer):
            msg = 'factorialk does not support non-integral scalar arguments!'
            raise ValueError(msg + helpmsg)
        elif n < 0:
            return 0
        elif n in {0, 1}:
            return 1
        return _range_prod(1, n, k=k)
    n = asarray(n)
    if n.size == 0:
        return n
    if not np.issubdtype(n.dtype, np.integer):
        msg = 'factorialk does not support non-integral arrays!'
        raise ValueError(msg + helpmsg)
    return _exact_factorialx_array(n, k=k)
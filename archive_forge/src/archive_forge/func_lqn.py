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
def lqn(n, z):
    """Legendre function of the second kind.

    Compute sequence of Legendre functions of the second kind, Qn(z) and
    derivatives for all degrees from 0 to n (inclusive).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError('arguments must be scalars.')
    n = _nonneg_int_or_fail(n, 'n', strict=False)
    if n < 1:
        n1 = 1
    else:
        n1 = n
    if iscomplex(z):
        qn, qd = _specfun.clqn(n1, z)
    else:
        qn, qd = _specfun.lqnb(n1, z)
    return (qn[:n + 1], qd[:n + 1])
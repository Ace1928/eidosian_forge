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
def lmbda(v, x):
    """Jahnke-Emden Lambda function, Lambdav(x).

    This function is defined as [2]_,

    .. math:: \\Lambda_v(x) = \\Gamma(v+1) \\frac{J_v(x)}{(x/2)^v},

    where :math:`\\Gamma` is the gamma function and :math:`J_v` is the
    Bessel function of the first kind.

    Parameters
    ----------
    v : float
        Order of the Lambda function
    x : float
        Value at which to evaluate the function and derivatives

    Returns
    -------
    vl : ndarray
        Values of Lambda_vi(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dl : ndarray
        Derivatives Lambda_vi'(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] Jahnke, E. and Emde, F. "Tables of Functions with Formulae and
           Curves" (4th ed.), Dover, 1945
    """
    if not (isscalar(v) and isscalar(x)):
        raise ValueError('arguments must be scalars.')
    if v < 0:
        raise ValueError('argument must be > 0.')
    n = int(v)
    v0 = v - n
    if n < 1:
        n1 = 1
    else:
        n1 = n
    v1 = n1 + v0
    if v != floor(v):
        vm, vl, dl = _specfun.lamv(v1, x)
    else:
        vm, vl, dl = _specfun.lamn(v1, x)
    return (vl[:n + 1], dl[:n + 1])
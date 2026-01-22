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
def assoc_laguerre(x, n, k=0.0):
    """Compute the generalized (associated) Laguerre polynomial of degree n and order k.

    The polynomial :math:`L^{(k)}_n(x)` is orthogonal over ``[0, inf)``,
    with weighting function ``exp(-x) * x**k`` with ``k > -1``.

    Parameters
    ----------
    x : float or ndarray
        Points where to evaluate the Laguerre polynomial
    n : int
        Degree of the Laguerre polynomial
    k : int
        Order of the Laguerre polynomial

    Returns
    -------
    assoc_laguerre: float or ndarray
        Associated laguerre polynomial values

    Notes
    -----
    `assoc_laguerre` is a simple wrapper around `eval_genlaguerre`, with
    reversed argument order ``(x, n, k=0.0) --> (n, k, x)``.

    """
    return _ufuncs.eval_genlaguerre(n, k, x)
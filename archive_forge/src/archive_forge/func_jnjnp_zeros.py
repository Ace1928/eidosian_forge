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
def jnjnp_zeros(nt):
    """Compute zeros of integer-order Bessel functions Jn and Jn'.

    Results are arranged in order of the magnitudes of the zeros.

    Parameters
    ----------
    nt : int
        Number (<=1200) of zeros to compute

    Returns
    -------
    zo[l-1] : ndarray
        Value of the lth zero of Jn(x) and Jn'(x). Of length `nt`.
    n[l-1] : ndarray
        Order of the Jn(x) or Jn'(x) associated with lth zero. Of length `nt`.
    m[l-1] : ndarray
        Serial number of the zeros of Jn(x) or Jn'(x) associated
        with lth zero. Of length `nt`.
    t[l-1] : ndarray
        0 if lth zero in zo is zero of Jn(x), 1 if it is a zero of Jn'(x). Of
        length `nt`.

    See Also
    --------
    jn_zeros, jnp_zeros : to get separated arrays of zeros.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or floor(nt) != nt or nt > 1200:
        raise ValueError('Number must be integer <= 1200.')
    nt = int(nt)
    n, m, t, zo = _specfun.jdzo(nt)
    return (zo[1:nt + 1], n[:nt], m[:nt], t[:nt])
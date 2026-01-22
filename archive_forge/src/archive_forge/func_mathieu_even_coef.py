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
def mathieu_even_coef(m, q):
    """Fourier coefficients for even Mathieu and modified Mathieu functions.

    The Fourier series of the even solutions of the Mathieu differential
    equation are of the form

    .. math:: \\mathrm{ce}_{2n}(z, q) = \\sum_{k=0}^{\\infty} A_{(2n)}^{(2k)} \\cos 2kz

    .. math:: \\mathrm{ce}_{2n+1}(z, q) = 
              \\sum_{k=0}^{\\infty} A_{(2n+1)}^{(2k+1)} \\cos (2k+1)z

    This function returns the coefficients :math:`A_{(2n)}^{(2k)}` for even
    input m=2n, and the coefficients :math:`A_{(2n+1)}^{(2k+1)}` for odd input
    m=2n+1.

    Parameters
    ----------
    m : int
        Order of Mathieu functions.  Must be non-negative.
    q : float (>=0)
        Parameter of Mathieu functions.  Must be non-negative.

    Returns
    -------
    Ak : ndarray
        Even or odd Fourier coefficients, corresponding to even or odd m.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/28.4#i

    """
    if not (isscalar(m) and isscalar(q)):
        raise ValueError('m and q must be scalars.')
    if q < 0:
        raise ValueError('q >=0')
    if m != floor(m) or m < 0:
        raise ValueError('m must be an integer >=0.')
    if q <= 1:
        qm = 7.5 + 56.1 * sqrt(q) - 134.7 * q + 90.7 * sqrt(q) * q
    else:
        qm = 17.0 + 3.1 * sqrt(q) - 0.126 * q + 0.0037 * sqrt(q) * q
    km = int(qm + 0.5 * m)
    if km > 251:
        warnings.warn('Too many predicted coefficients.', RuntimeWarning, stacklevel=2)
    kd = 1
    m = int(floor(m))
    if m % 2:
        kd = 2
    a = mathieu_a(m, q)
    fc = _specfun.fcoef(kd, m, q, a)
    return fc[:km]
import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def chebyc(n, monic=False):
    """Chebyshev polynomial of the first kind on :math:`[-2, 2]`.

    Defined as :math:`C_n(x) = 2T_n(x/2)`, where :math:`T_n` is the
    nth Chebychev polynomial of the first kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    C : orthopoly1d
        Chebyshev polynomial of the first kind on :math:`[-2, 2]`.

    See Also
    --------
    chebyt : Chebyshev polynomial of the first kind.

    Notes
    -----
    The polynomials :math:`C_n(x)` are orthogonal over :math:`[-2, 2]`
    with weight function :math:`1/\\sqrt{1 - (x/2)^2}`.

    References
    ----------
    .. [1] Abramowitz and Stegun, "Handbook of Mathematical Functions"
           Section 22. National Bureau of Standards, 1972.

    """
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_chebyc(n1)
    if n == 0:
        x, w = ([], [])
    hn = 4 * pi * ((n == 0) + 1)
    kn = 1.0
    p = orthopoly1d(x, w, hn, kn, wfunc=lambda x: 1.0 / sqrt(1 - x * x / 4.0), limits=(-2, 2), monic=monic)
    if not monic:
        p._scale(2.0 / p(2))
        p.__dict__['_eval_func'] = lambda x: _ufuncs.eval_chebyc(n, x)
    return p
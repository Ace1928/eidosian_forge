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
def yvp(v, z, n=1):
    """Compute derivatives of Bessel functions of the second kind.

    Compute the nth derivative of the Bessel function `Yv` with
    respect to `z`.

    Parameters
    ----------
    v : array_like of float
        Order of Bessel function
    z : complex
        Argument at which to evaluate the derivative
    n : int, default 1
        Order of derivative. For 0 returns the BEssel function `yv`

    Returns
    -------
    scalar or ndarray
        nth derivative of the Bessel function.

    See Also
    --------
    yv : Bessel functions of the second kind

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    Examples
    --------
    Compute the Bessel function of the second kind of order 0 and
    its first two derivatives at 1.

    >>> from scipy.special import yvp
    >>> yvp(0, 1, 0), yvp(0, 1, 1), yvp(0, 1, 2)
    (0.088256964215677, 0.7812128213002889, -0.8694697855159659)

    Compute the first derivative of the Bessel function of the second
    kind for several orders at 1 by providing an array for `v`.

    >>> yvp([0, 1, 2], 1, 1)
    array([0.78121282, 0.86946979, 2.52015239])

    Compute the first derivative of the Bessel function of the
    second kind of order 0 at several points by providing an array for `z`.

    >>> import numpy as np
    >>> points = np.array([0.5, 1.5, 3.])
    >>> yvp(0, points, 1)
    array([ 1.47147239,  0.41230863, -0.32467442])

    Plot the Bessel function of the second kind of order 1 and its
    first three derivatives.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 5, 1000)
    >>> x[0] += 1e-15
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, yvp(1, x, 0), label=r"$Y_1$")
    >>> ax.plot(x, yvp(1, x, 1), label=r"$Y_1'$")
    >>> ax.plot(x, yvp(1, x, 2), label=r"$Y_1''$")
    >>> ax.plot(x, yvp(1, x, 3), label=r"$Y_1'''$")
    >>> ax.set_ylim(-10, 10)
    >>> plt.legend()
    >>> plt.show()
    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return yv(v, z)
    else:
        return _bessel_diff_formula(v, z, n, yv, -1)
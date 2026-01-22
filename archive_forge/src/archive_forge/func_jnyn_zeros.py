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
def jnyn_zeros(n, nt):
    """Compute nt zeros of Bessel functions Jn(x), Jn'(x), Yn(x), and Yn'(x).

    Returns 4 arrays of length `nt`, corresponding to the first `nt`
    zeros of Jn(x), Jn'(x), Yn(x), and Yn'(x), respectively. The zeros
    are returned in ascending order.

    Parameters
    ----------
    n : int
        Order of the Bessel functions
    nt : int
        Number (<=1200) of zeros to compute

    Returns
    -------
    Jn : ndarray
        First `nt` zeros of Jn
    Jnp : ndarray
        First `nt` zeros of Jn'
    Yn : ndarray
        First `nt` zeros of Yn
    Ynp : ndarray
        First `nt` zeros of Yn'

    See Also
    --------
    jn_zeros, jnp_zeros, yn_zeros, ynp_zeros

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first three roots of :math:`J_1`, :math:`J_1'`,
    :math:`Y_1` and :math:`Y_1'`.

    >>> from scipy.special import jnyn_zeros
    >>> jn_roots, jnp_roots, yn_roots, ynp_roots = jnyn_zeros(1, 3)
    >>> jn_roots, yn_roots
    (array([ 3.83170597,  7.01558667, 10.17346814]),
     array([2.19714133, 5.42968104, 8.59600587]))

    Plot :math:`J_1`, :math:`J_1'`, :math:`Y_1`, :math:`Y_1'` and their roots.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import jnyn_zeros, jvp, jn, yvp, yn
    >>> jn_roots, jnp_roots, yn_roots, ynp_roots = jnyn_zeros(1, 3)
    >>> fig, ax = plt.subplots()
    >>> xmax= 11
    >>> x = np.linspace(0, xmax)
    >>> x[0] += 1e-15
    >>> ax.plot(x, jn(1, x), label=r"$J_1$", c='r')
    >>> ax.plot(x, jvp(1, x, 1), label=r"$J_1'$", c='b')
    >>> ax.plot(x, yn(1, x), label=r"$Y_1$", c='y')
    >>> ax.plot(x, yvp(1, x, 1), label=r"$Y_1'$", c='c')
    >>> zeros = np.zeros((3, ))
    >>> ax.scatter(jn_roots, zeros, s=30, c='r', zorder=5,
    ...            label=r"$J_1$ roots")
    >>> ax.scatter(jnp_roots, zeros, s=30, c='b', zorder=5,
    ...            label=r"$J_1'$ roots")
    >>> ax.scatter(yn_roots, zeros, s=30, c='y', zorder=5,
    ...            label=r"$Y_1$ roots")
    >>> ax.scatter(ynp_roots, zeros, s=30, c='c', zorder=5,
    ...            label=r"$Y_1'$ roots")
    >>> ax.hlines(0, 0, xmax, color='k')
    >>> ax.set_ylim(-0.6, 0.6)
    >>> ax.set_xlim(0, xmax)
    >>> ax.legend(ncol=2, bbox_to_anchor=(1., 0.75))
    >>> plt.tight_layout()
    >>> plt.show()
    """
    if not (isscalar(nt) and isscalar(n)):
        raise ValueError('Arguments must be scalars.')
    if floor(n) != n or floor(nt) != nt:
        raise ValueError('Arguments must be integers.')
    if nt <= 0:
        raise ValueError('nt > 0')
    return _specfun.jyzo(abs(n), nt)
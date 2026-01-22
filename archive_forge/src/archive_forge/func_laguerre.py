import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def laguerre(n, monic=False):
    """Laguerre polynomial.

    Defined to be the solution of

    .. math::
        x\\frac{d^2}{dx^2}L_n + (1 - x)\\frac{d}{dx}L_n + nL_n = 0;

    :math:`L_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    L : orthopoly1d
        Laguerre Polynomial.

    See Also
    --------
    genlaguerre : Generalized (associated) Laguerre polynomial.

    Notes
    -----
    The polynomials :math:`L_n` are orthogonal over :math:`[0,
    \\infty)` with weight function :math:`e^{-x}`.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The Laguerre polynomials :math:`L_n` are the special case
    :math:`\\alpha = 0` of the generalized Laguerre polynomials
    :math:`L_n^{(\\alpha)}`.
    Let's verify it on the interval :math:`[-1, 1]`:

    >>> import numpy as np
    >>> from scipy.special import genlaguerre
    >>> from scipy.special import laguerre
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> np.allclose(genlaguerre(3, 0)(x), laguerre(3)(x))
    True

    The polynomials :math:`L_n` also satisfy the recurrence relation:

    .. math::
        (n + 1)L_{n+1}(x) = (2n +1 -x)L_n(x) - nL_{n-1}(x)

    This can be easily checked on :math:`[0, 1]` for :math:`n = 3`:

    >>> x = np.arange(0.0, 1.0, 0.01)
    >>> np.allclose(4 * laguerre(4)(x),
    ...             (7 - x) * laguerre(3)(x) - 3 * laguerre(2)(x))
    True

    This is the plot of the first few Laguerre polynomials :math:`L_n`:

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(-1.0, 5.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-5.0, 5.0)
    >>> ax.set_title(r'Laguerre polynomials $L_n$')
    >>> for n in np.arange(0, 5):
    ...     ax.plot(x, laguerre(n)(x), label=rf'$L_{n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_laguerre(n1)
    if n == 0:
        x, w = ([], [])
    hn = 1.0
    kn = (-1) ** n / _gam(n + 1)
    p = orthopoly1d(x, w, hn, kn, lambda x: exp(-x), (0, inf), monic, lambda x: _ufuncs.eval_laguerre(n, x))
    return p
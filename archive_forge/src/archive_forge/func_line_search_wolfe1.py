from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np
def line_search_wolfe1(f, fprime, xk, pk, gfk=None, old_fval=None, old_old_fval=None, args=(), c1=0.0001, c2=0.9, amax=50, amin=1e-08, xtol=1e-14):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`

    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`

    The rest of the parameters are the same as for `scalar_search_wolfe1`.

    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.

    """
    if gfk is None:
        gfk = fprime(xk, *args)
    gval = [gfk]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        return f(xk + s * pk, *args)

    def derphi(s):
        gval[0] = fprime(xk + s * pk, *args)
        gc[0] += 1
        return np.dot(gval[0], pk)
    derphi0 = np.dot(gfk, pk)
    stp, fval, old_fval = scalar_search_wolfe1(phi, derphi, old_fval, old_old_fval, derphi0, c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)
    return (stp, fc[0], gc[0], fval, old_fval, gval[0])
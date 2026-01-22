from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np
def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, derphi0=None, c1=0.0001, c2=0.9, amax=50, amin=1e-08, xtol=1e-14):
    """
    Scalar function search for alpha that satisfies strong Wolfe conditions

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax, amin : float, optional
        Maximum and minimum step size
    xtol : float, optional
        Relative tolerance for an acceptable step.

    Returns
    -------
    alpha : float
        Step size, or None if no suitable step was found
    phi : float
        Value of `phi` at the new point `alpha`
    phi0 : float
        Value of `phi` at `alpha=0`

    Notes
    -----
    Uses routine DCSRCH from MINPACK.
    
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1`` as described in [1]_.

    References
    ----------
    
    .. [1] Nocedal, J., & Wright, S. J. (2006). Numerical optimization.
       In Springer Series in Operations Research and Financial Engineering.
       (Springer Series in Operations Research and Financial Engineering).
       Springer Nature.

    """
    _check_c1_c2(c1, c2)
    if phi0 is None:
        phi0 = phi(0.0)
    if derphi0 is None:
        derphi0 = derphi(0.0)
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0
    maxiter = 100
    dcsrch = DCSRCH(phi, derphi, c1, c2, xtol, amin, amax)
    stp, phi1, phi0, task = dcsrch(alpha1, phi0=phi0, derphi0=derphi0, maxiter=maxiter)
    return (stp, phi1, phi0)
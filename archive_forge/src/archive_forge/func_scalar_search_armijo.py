from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np
def scalar_search_armijo(phi, phi0, derphi0, c1=0.0001, alpha0=1, amin=0):
    """Minimize over alpha, the function ``phi(alpha)``.

    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    alpha > 0 is assumed to be a descent direction.

    Returns
    -------
    alpha
    phi1

    """
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return (alpha0, phi_a0)
    alpha1 = -derphi0 * alpha0 ** 2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)
    if phi_a1 <= phi0 + c1 * alpha1 * derphi0:
        return (alpha1, phi_a1)
    while alpha1 > amin:
        factor = alpha0 ** 2 * alpha1 ** 2 * (alpha1 - alpha0)
        a = alpha0 ** 2 * (phi_a1 - phi0 - derphi0 * alpha1) - alpha1 ** 2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -alpha0 ** 3 * (phi_a1 - phi0 - derphi0 * alpha1) + alpha1 ** 3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor
        alpha2 = (-b + np.sqrt(abs(b ** 2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)
        if phi_a2 <= phi0 + c1 * alpha2 * derphi0:
            return (alpha2, phi_a2)
        if alpha1 - alpha2 > alpha1 / 2.0 or 1 - alpha2 / alpha1 < 0.96:
            alpha2 = alpha1 / 2.0
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2
    return (None, phi_a1)
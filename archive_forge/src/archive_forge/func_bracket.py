import math
import warnings
import sys
import inspect
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
import numpy as np
from scipy.linalg import cholesky, issymmetric, LinAlgError
from scipy.sparse.linalg import LinearOperator
from ._linesearch import (line_search_wolfe1, line_search_wolfe2,
from ._numdiff import approx_derivative
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy._lib._util import MapWrapper, check_random_state
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
def bracket(func, xa=0.0, xb=1.0, args=(), grow_limit=110.0, maxiter=1000):
    """
    Bracket the minimum of a function.

    Given a function and distinct initial points, search in the
    downhill direction (as defined by the initial points) and return
    three points that bracket the minimum of the function.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to minimize.
    xa, xb : float, optional
        Initial points. Defaults `xa` to 0.0, and `xb` to 1.0.
        A local minimum need not be contained within this interval.
    args : tuple, optional
        Additional arguments (if present), passed to `func`.
    grow_limit : float, optional
        Maximum grow limit.  Defaults to 110.0
    maxiter : int, optional
        Maximum number of iterations to perform. Defaults to 1000.

    Returns
    -------
    xa, xb, xc : float
        Final points of the bracket.
    fa, fb, fc : float
        Objective function values at the bracket points.
    funcalls : int
        Number of function evaluations made.

    Raises
    ------
    BracketError
        If no valid bracket is found before the algorithm terminates.
        See notes for conditions of a valid bracket.

    Notes
    -----
    The algorithm attempts to find three strictly ordered points (i.e.
    :math:`x_a < x_b < x_c` or :math:`x_c < x_b < x_a`) satisfying
    :math:`f(x_b) ≤ f(x_a)` and :math:`f(x_b) ≤ f(x_c)`, where one of the
    inequalities must be satistfied strictly and all :math:`x_i` must be
    finite.

    Examples
    --------
    This function can find a downward convex region of a function:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.optimize import bracket
    >>> def f(x):
    ...     return 10*x**2 + 3*x + 5
    >>> x = np.linspace(-2, 2)
    >>> y = f(x)
    >>> init_xa, init_xb = 0.1, 1
    >>> xa, xb, xc, fa, fb, fc, funcalls = bracket(f, xa=init_xa, xb=init_xb)
    >>> plt.axvline(x=init_xa, color="k", linestyle="--")
    >>> plt.axvline(x=init_xb, color="k", linestyle="--")
    >>> plt.plot(x, y, "-k")
    >>> plt.plot(xa, fa, "bx")
    >>> plt.plot(xb, fb, "rx")
    >>> plt.plot(xc, fc, "bx")
    >>> plt.show()

    Note that both initial points were to the right of the minimum, and the
    third point was found in the "downhill" direction: the direction
    in which the function appeared to be decreasing (to the left).
    The final points are strictly ordered, and the function value
    at the middle point is less than the function values at the endpoints;
    it follows that a minimum must lie within the bracket.

    """
    _gold = 1.618034
    _verysmall_num = 1e-21
    xa, xb = np.asarray([xa, xb])
    fa = func(*(xa,) + args)
    fb = func(*(xb,) + args)
    if fa < fb:
        xa, xb = (xb, xa)
        fa, fb = (fb, fa)
    xc = xb + _gold * (xb - xa)
    fc = func(*(xc,) + args)
    funcalls = 3
    iter = 0
    while fc < fb:
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        if np.abs(val) < _verysmall_num:
            denom = 2.0 * _verysmall_num
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        msg = 'No valid bracket was found before the iteration limit was reached. Consider trying different initial points or increasing `maxiter`.'
        if iter > maxiter:
            raise RuntimeError(msg)
        iter += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = func(*(w,) + args)
            funcalls += 1
            if fw < fc:
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif fw > fb:
                xc = w
                fc = fw
                break
            w = xc + _gold * (xc - xb)
            fw = func(*(w,) + args)
            funcalls += 1
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw = func(*(w,) + args)
            funcalls += 1
        elif (w - wlim) * (xc - w) > 0.0:
            fw = func(*(w,) + args)
            funcalls += 1
            if fw < fc:
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = func(*(w,) + args)
                funcalls += 1
        else:
            w = xc + _gold * (xc - xb)
            fw = func(*(w,) + args)
            funcalls += 1
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw
    cond1 = fb < fc and fb <= fa or (fb < fa and fb <= fc)
    cond2 = xa < xb < xc or xc < xb < xa
    cond3 = np.isfinite(xa) and np.isfinite(xb) and np.isfinite(xc)
    msg = 'The algorithm terminated without finding a valid bracket. Consider trying different initial points.'
    if not (cond1 and cond2 and cond3):
        e = BracketError(msg)
        e.data = (xa, xb, xc, fa, fb, fc, funcalls)
        raise e
    return (xa, xb, xc, fa, fb, fc, funcalls)
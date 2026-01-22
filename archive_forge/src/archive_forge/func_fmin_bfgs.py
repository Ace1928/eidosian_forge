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
def fmin_bfgs(f, x0, fprime=None, args=(), gtol=1e-05, norm=np.inf, epsilon=_epsilon, maxiter=None, full_output=0, disp=1, retall=0, callback=None, xrtol=0, c1=0.0001, c2=0.9, hess_inv0=None):
    """
    Minimize a function using the BFGS algorithm.

    Parameters
    ----------
    f : callable ``f(x,*args)``
        Objective function to be minimized.
    x0 : ndarray
        Initial guess, shape (n,)
    fprime : callable ``f'(x,*args)``, optional
        Gradient of f.
    args : tuple, optional
        Extra arguments passed to f and fprime.
    gtol : float, optional
        Terminate successfully if gradient norm is less than `gtol`
    norm : float, optional
        Order of norm (Inf is max, -Inf is min)
    epsilon : int or ndarray, optional
        If `fprime` is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function to call after each
        iteration. Called as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True, return ``fopt``, ``func_calls``, ``grad_calls``, and
        ``warnflag`` in addition to ``xopt``.
    disp : bool, optional
        Print convergence message if True.
    retall : bool, optional
        Return a list of results at each iteration if True.
    xrtol : float, default: 0
        Relative tolerance for `x`. Terminate successfully if step
        size is less than ``xk * xrtol`` where ``xk`` is the current
        parameter vector.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.9
        Parameter for curvature condition rule.
    hess_inv0 : None or ndarray, optional``
        Initial inverse hessian estimate, shape (n, n). If None (default) then
        the identity matrix is used.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e., ``f(xopt) == fopt``.
    fopt : float
        Minimum value.
    gopt : ndarray
        Value of gradient at minimum, f'(xopt), which should be near 0.
    Bopt : ndarray
        Value of 1/f''(xopt), i.e., the inverse Hessian matrix.
    func_calls : int
        Number of function_calls made.
    grad_calls : int
        Number of gradient calls made.
    warnflag : integer
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
        3 : NaN result encountered.
    allvecs : list
        The value of `xopt` at each iteration. Only returned if `retall` is
        True.

    Notes
    -----
    Optimize the function, `f`, whose gradient is given by `fprime`
    using the quasi-Newton method of Broyden, Fletcher, Goldfarb,
    and Shanno (BFGS).
    
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.

    See Also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See ``method='BFGS'`` in particular.

    References
    ----------
    Wright, and Nocedal 'Numerical Optimization', 1999, p. 198.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import fmin_bfgs
    >>> def quadratic_cost(x, Q):
    ...     return x @ Q @ x
    ...
    >>> x0 = np.array([-3, -4])
    >>> cost_weight =  np.diag([1., 10.])
    >>> # Note that a trailing comma is necessary for a tuple with single element
    >>> fmin_bfgs(quadratic_cost, x0, args=(cost_weight,))
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 7                   # may vary
            Function evaluations: 24        # may vary
            Gradient evaluations: 8         # may vary
    array([ 2.85169950e-06, -4.61820139e-07])

    >>> def quadratic_cost_grad(x, Q):
    ...     return 2 * Q @ x
    ...
    >>> fmin_bfgs(quadratic_cost, x0, quadratic_cost_grad, args=(cost_weight,))
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 7
            Function evaluations: 8
            Gradient evaluations: 8
    array([ 2.85916637e-06, -4.54371951e-07])

    """
    opts = {'gtol': gtol, 'norm': norm, 'eps': epsilon, 'disp': disp, 'maxiter': maxiter, 'return_all': retall, 'xrtol': xrtol, 'c1': c1, 'c2': c2, 'hess_inv0': hess_inv0}
    callback = _wrap_callback(callback)
    res = _minimize_bfgs(f, x0, args, fprime, callback=callback, **opts)
    if full_output:
        retlist = (res['x'], res['fun'], res['jac'], res['hess_inv'], res['nfev'], res['njev'], res['status'])
        if retall:
            retlist += (res['allvecs'],)
        return retlist
    elif retall:
        return (res['x'], res['allvecs'])
    else:
        return res['x']
import numpy as np
from . import _zeros_py as optzeros
from ._numdiff import approx_derivative
def root_scalar(f, args=(), method=None, bracket=None, fprime=None, fprime2=None, x0=None, x1=None, xtol=None, rtol=None, maxiter=None, options=None):
    """
    Find a root of a scalar function.

    Parameters
    ----------
    f : callable
        A function to find a root of.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative(s).
    method : str, optional
        Type of solver.  Should be one of

            - 'bisect'    :ref:`(see here) <optimize.root_scalar-bisect>`
            - 'brentq'    :ref:`(see here) <optimize.root_scalar-brentq>`
            - 'brenth'    :ref:`(see here) <optimize.root_scalar-brenth>`
            - 'ridder'    :ref:`(see here) <optimize.root_scalar-ridder>`
            - 'toms748'    :ref:`(see here) <optimize.root_scalar-toms748>`
            - 'newton'    :ref:`(see here) <optimize.root_scalar-newton>`
            - 'secant'    :ref:`(see here) <optimize.root_scalar-secant>`
            - 'halley'    :ref:`(see here) <optimize.root_scalar-halley>`

    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  `f(x, *args)` must have different
        signs at the two endpoints.
    x0 : float, optional
        Initial guess.
    x1 : float, optional
        A second guess.
    fprime : bool or callable, optional
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the derivative.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    fprime2 : bool or callable, optional
        If `fprime2` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the
        first and second derivatives.
        `fprime2` can also be a callable returning the second derivative of `f`.
        In this case, it must accept the same arguments as `f`.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options : dict, optional
        A dictionary of solver options. E.g., ``k``, see
        :obj:`show_options()` for details.

    Returns
    -------
    sol : RootResults
        The solution represented as a ``RootResults`` object.
        Important attributes are: ``root`` the solution , ``converged`` a
        boolean flag indicating if the algorithm exited successfully and
        ``flag`` which describes the cause of the termination. See
        `RootResults` for a description of other attributes.

    See also
    --------
    show_options : Additional options accepted by the solvers
    root : Find a root of a vector function.

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter.

    The default is to use the best method available for the situation
    presented.
    If a bracket is provided, it may use one of the bracketing methods.
    If a derivative and an initial value are specified, it may
    select one of the derivative-based methods.
    If no method is judged applicable, it will raise an Exception.

    Arguments for each method are as follows (x=required, o=optional).

    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    |                    method                     | f | args | bracket | x0 | x1 | fprime | fprime2 | xtol | rtol | maxiter | options |
    +===============================================+===+======+=========+====+====+========+=========+======+======+=========+=========+
    | :ref:`bisect <optimize.root_scalar-bisect>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`brentq <optimize.root_scalar-brentq>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`brenth <optimize.root_scalar-brenth>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`ridder <optimize.root_scalar-ridder>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`toms748 <optimize.root_scalar-toms748>` | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`secant <optimize.root_scalar-secant>`   | x |  o   |         | x  | o  |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`newton <optimize.root_scalar-newton>`   | x |  o   |         | x  |    |   o    |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`halley <optimize.root_scalar-halley>`   | x |  o   |         | x  |    |   x    |    x    |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+

    Examples
    --------

    Find the root of a simple cubic

    >>> from scipy import optimize
    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1

    >>> def fprime(x):
    ...     return 3*x**2

    The `brentq` method takes as input a bracket

    >>> sol = optimize.root_scalar(f, bracket=[0, 3], method='brentq')
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 10, 11)

    The `newton` method takes as input a single point and uses the
    derivative(s).

    >>> sol = optimize.root_scalar(f, x0=0.2, fprime=fprime, method='newton')
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 11, 22)

    The function can provide the value and derivative(s) in a single call.

    >>> def f_p_pp(x):
    ...     return (x**3 - 1), 3*x**2, 6*x

    >>> sol = optimize.root_scalar(
    ...     f_p_pp, x0=0.2, fprime=True, method='newton'
    ... )
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 11, 11)

    >>> sol = optimize.root_scalar(
    ...     f_p_pp, x0=0.2, fprime=True, fprime2=True, method='halley'
    ... )
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 7, 8)


    """
    if not isinstance(args, tuple):
        args = (args,)
    if options is None:
        options = {}
    is_memoized = False
    if fprime2 is not None and (not callable(fprime2)):
        if bool(fprime2):
            f = MemoizeDer(f)
            is_memoized = True
            fprime2 = f.fprime2
            fprime = f.fprime
        else:
            fprime2 = None
    if fprime is not None and (not callable(fprime)):
        if bool(fprime):
            f = MemoizeDer(f)
            is_memoized = True
            fprime = f.fprime
        else:
            fprime = None
    kwargs = {}
    for k in ['xtol', 'rtol', 'maxiter']:
        v = locals().get(k)
        if v is not None:
            kwargs[k] = v
    if options:
        kwargs.update(options)
    kwargs.update(full_output=True, disp=False)
    if not method:
        if bracket:
            method = 'brentq'
        elif x0 is not None:
            if fprime:
                if fprime2:
                    method = 'halley'
                else:
                    method = 'newton'
            elif x1 is not None:
                method = 'secant'
            else:
                method = 'newton'
    if not method:
        raise ValueError('Unable to select a solver as neither bracket nor starting point provided.')
    meth = method.lower()
    map2underlying = {'halley': 'newton', 'secant': 'newton'}
    try:
        methodc = getattr(optzeros, map2underlying.get(meth, meth))
    except AttributeError as e:
        raise ValueError('Unknown solver %s' % meth) from e
    if meth in ['bisect', 'ridder', 'brentq', 'brenth', 'toms748']:
        if not isinstance(bracket, (list, tuple, np.ndarray)):
            raise ValueError('Bracket needed for %s' % method)
        a, b = bracket[:2]
        try:
            r, sol = methodc(f, a, b, args=args, **kwargs)
        except ValueError as e:
            if hasattr(e, '_x'):
                sol = optzeros.RootResults(root=e._x, iterations=np.nan, function_calls=e._function_calls, flag=str(e), method=method)
            else:
                raise
    elif meth in ['secant']:
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=None, fprime2=None, x1=x1, **kwargs)
    elif meth in ['newton']:
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        if not fprime:

            def fprime(x, *args):
                return approx_derivative(f, x, method='2-point', args=args)[0]
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=fprime, fprime2=None, **kwargs)
    elif meth in ['halley']:
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        if not fprime:
            raise ValueError('fprime must be specified for %s' % method)
        if not fprime2:
            raise ValueError('fprime2 must be specified for %s' % method)
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=fprime, fprime2=fprime2, **kwargs)
    else:
        raise ValueError('Unknown solver %s' % method)
    if is_memoized:
        n_calls = f.n_calls
        sol.function_calls = n_calls
    return sol
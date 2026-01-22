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
def brute(func, ranges, args=(), Ns=20, full_output=0, finish=fmin, disp=False, workers=1):
    """Minimize a function over a given range by brute force.

    Uses the "brute force" method, i.e., computes the function's value
    at each point of a multidimensional grid of points, to find the global
    minimum of the function.

    The function is evaluated everywhere in the range with the datatype of the
    first call to the function, as enforced by the ``vectorize`` NumPy
    function. The value and type of the function evaluation returned when
    ``full_output=True`` are affected in addition by the ``finish`` argument
    (see Notes).

    The brute force approach is inefficient because the number of grid points
    increases exponentially - the number of grid points to evaluate is
    ``Ns ** len(x)``. Consequently, even with coarse grid spacing, even
    moderately sized problems can take a long time to run, and/or run into
    memory limitations.

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the
        form ``f(x, *args)``, where ``x`` is the argument in
        the form of a 1-D array and ``args`` is a tuple of any
        additional fixed parameters needed to completely specify
        the function.
    ranges : tuple
        Each component of the `ranges` tuple must be either a
        "slice object" or a range tuple of the form ``(low, high)``.
        The program uses these to create the grid of points on which
        the objective function will be computed. See `Note 2` for
        more detail.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify
        the function.
    Ns : int, optional
        Number of grid points along the axes, if not otherwise
        specified. See `Note2`.
    full_output : bool, optional
        If True, return the evaluation grid and the objective function's
        values on it.
    finish : callable, optional
        An optimization function that is called with the result of brute force
        minimization as initial guess. `finish` should take `func` and
        the initial guess as positional arguments, and take `args` as
        keyword arguments. It may additionally take `full_output`
        and/or `disp` as keyword arguments. Use None if no "polishing"
        function is to be used. See Notes for more details.
    disp : bool, optional
        Set to True to print convergence messages from the `finish` callable.
    workers : int or map-like callable, optional
        If `workers` is an int the grid is subdivided into `workers`
        sections and evaluated in parallel (uses
        `multiprocessing.Pool <multiprocessing>`).
        Supply `-1` to use all cores available to the Process.
        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for evaluating the grid in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
        Requires that `func` be pickleable.

        .. versionadded:: 1.3.0

    Returns
    -------
    x0 : ndarray
        A 1-D array containing the coordinates of a point at which the
        objective function had its minimum value. (See `Note 1` for
        which point is returned.)
    fval : float
        Function value at the point `x0`. (Returned when `full_output` is
        True.)
    grid : tuple
        Representation of the evaluation grid. It has the same
        length as `x0`. (Returned when `full_output` is True.)
    Jout : ndarray
        Function values at each point of the evaluation
        grid, i.e., ``Jout = func(*grid)``. (Returned
        when `full_output` is True.)

    See Also
    --------
    basinhopping, differential_evolution

    Notes
    -----
    *Note 1*: The program finds the gridpoint at which the lowest value
    of the objective function occurs. If `finish` is None, that is the
    point returned. When the global minimum occurs within (or not very far
    outside) the grid's boundaries, and the grid is fine enough, that
    point will be in the neighborhood of the global minimum.

    However, users often employ some other optimization program to
    "polish" the gridpoint values, i.e., to seek a more precise
    (local) minimum near `brute's` best gridpoint.
    The `brute` function's `finish` option provides a convenient way to do
    that. Any polishing program used must take `brute's` output as its
    initial guess as a positional argument, and take `brute's` input values
    for `args` as keyword arguments, otherwise an error will be raised.
    It may additionally take `full_output` and/or `disp` as keyword arguments.

    `brute` assumes that the `finish` function returns either an
    `OptimizeResult` object or a tuple in the form:
    ``(xmin, Jmin, ... , statuscode)``, where ``xmin`` is the minimizing
    value of the argument, ``Jmin`` is the minimum value of the objective
    function, "..." may be some other returned values (which are not used
    by `brute`), and ``statuscode`` is the status code of the `finish` program.

    Note that when `finish` is not None, the values returned are those
    of the `finish` program, *not* the gridpoint ones. Consequently,
    while `brute` confines its search to the input grid points,
    the `finish` program's results usually will not coincide with any
    gridpoint, and may fall outside the grid's boundary. Thus, if a
    minimum only needs to be found over the provided grid points, make
    sure to pass in `finish=None`.

    *Note 2*: The grid of points is a `numpy.mgrid` object.
    For `brute` the `ranges` and `Ns` inputs have the following effect.
    Each component of the `ranges` tuple can be either a slice object or a
    two-tuple giving a range of values, such as (0, 5). If the component is a
    slice object, `brute` uses it directly. If the component is a two-tuple
    range, `brute` internally converts it to a slice object that interpolates
    `Ns` points from its low-value to its high-value, inclusive.

    Examples
    --------
    We illustrate the use of `brute` to seek the global minimum of a function
    of two variables that is given as the sum of a positive-definite
    quadratic and two deep "Gaussian-shaped" craters. Specifically, define
    the objective function `f` as the sum of three other functions,
    ``f = f1 + f2 + f3``. We suppose each of these has a signature
    ``(z, *params)``, where ``z = (x, y)``,  and ``params`` and the functions
    are as defined below.

    >>> import numpy as np
    >>> params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)
    >>> def f1(z, *params):
    ...     x, y = z
    ...     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    ...     return (a * x**2 + b * x * y + c * y**2 + d*x + e*y + f)

    >>> def f2(z, *params):
    ...     x, y = z
    ...     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    ...     return (-g*np.exp(-((x-h)**2 + (y-i)**2) / scale))

    >>> def f3(z, *params):
    ...     x, y = z
    ...     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    ...     return (-j*np.exp(-((x-k)**2 + (y-l)**2) / scale))

    >>> def f(z, *params):
    ...     return f1(z, *params) + f2(z, *params) + f3(z, *params)

    Thus, the objective function may have local minima near the minimum
    of each of the three functions of which it is composed. To
    use `fmin` to polish its gridpoint result, we may then continue as
    follows:

    >>> rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
    >>> from scipy import optimize
    >>> resbrute = optimize.brute(f, rranges, args=params, full_output=True,
    ...                           finish=optimize.fmin)
    >>> resbrute[0]  # global minimum
    array([-1.05665192,  1.80834843])
    >>> resbrute[1]  # function value at global minimum
    -3.4085818767

    Note that if `finish` had been set to None, we would have gotten the
    gridpoint [-1.0 1.75] where the rounded function value is -2.892.

    """
    N = len(ranges)
    if N > 40:
        raise ValueError('Brute Force not possible with more than 40 variables.')
    lrange = list(ranges)
    for k in range(N):
        if not isinstance(lrange[k], slice):
            if len(lrange[k]) < 3:
                lrange[k] = tuple(lrange[k]) + (complex(Ns),)
            lrange[k] = slice(*lrange[k])
    if N == 1:
        lrange = lrange[0]
    grid = np.mgrid[lrange]
    inpt_shape = grid.shape
    if N > 1:
        grid = np.reshape(grid, (inpt_shape[0], np.prod(inpt_shape[1:]))).T
    if not np.iterable(args):
        args = (args,)
    wrapped_func = _Brute_Wrapper(func, args)
    with MapWrapper(pool=workers) as mapper:
        Jout = np.array(list(mapper(wrapped_func, grid)))
        if N == 1:
            grid = (grid,)
            Jout = np.squeeze(Jout)
        elif N > 1:
            Jout = np.reshape(Jout, inpt_shape[1:])
            grid = np.reshape(grid.T, inpt_shape)
    Nshape = shape(Jout)
    indx = argmin(Jout.ravel(), axis=-1)
    Nindx = np.empty(N, int)
    xmin = np.empty(N, float)
    for k in range(N - 1, -1, -1):
        thisN = Nshape[k]
        Nindx[k] = indx % Nshape[k]
        indx = indx // thisN
    for k in range(N):
        xmin[k] = grid[k][tuple(Nindx)]
    Jmin = Jout[tuple(Nindx)]
    if N == 1:
        grid = grid[0]
        xmin = xmin[0]
    if callable(finish):
        finish_args = _getfullargspec(finish).args
        finish_kwargs = dict()
        if 'full_output' in finish_args:
            finish_kwargs['full_output'] = 1
        if 'disp' in finish_args:
            finish_kwargs['disp'] = disp
        elif 'options' in finish_args:
            finish_kwargs['options'] = {'disp': disp}
        res = finish(func, xmin, args=args, **finish_kwargs)
        if isinstance(res, OptimizeResult):
            xmin = res.x
            Jmin = res.fun
            success = res.success
        else:
            xmin = res[0]
            Jmin = res[1]
            success = res[-1] == 0
        if not success:
            if disp:
                warnings.warn('Either final optimization did not succeed or `finish` does not return `statuscode` as its last argument.', RuntimeWarning, stacklevel=2)
    if full_output:
        return (xmin, Jmin, grid, Jout)
    else:
        return xmin
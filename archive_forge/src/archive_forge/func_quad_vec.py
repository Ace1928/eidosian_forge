import sys
import copy
import heapq
import collections
import functools
import numpy as np
from scipy._lib._util import MapWrapper, _FunctionWrapper
def quad_vec(f, a, b, epsabs=1e-200, epsrel=1e-08, norm='2', cache_size=100000000.0, limit=10000, workers=1, points=None, quadrature=None, full_output=False, *, args=()):
    """Adaptive integration of a vector-valued function.

    Parameters
    ----------
    f : callable
        Vector-valued function f(x) to integrate.
    a : float
        Initial point.
    b : float
        Final point.
    epsabs : float, optional
        Absolute tolerance.
    epsrel : float, optional
        Relative tolerance.
    norm : {'max', '2'}, optional
        Vector norm to use for error estimation.
    cache_size : int, optional
        Number of bytes to use for memoization.
    limit : float or int, optional
        An upper bound on the number of subintervals used in the adaptive
        algorithm.
    workers : int or map-like callable, optional
        If `workers` is an integer, part of the computation is done in
        parallel subdivided to this many tasks (using
        :class:`python:multiprocessing.pool.Pool`).
        Supply `-1` to use all cores available to the Process.
        Alternatively, supply a map-like callable, such as
        :meth:`python:multiprocessing.pool.Pool.map` for evaluating the
        population in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
    points : list, optional
        List of additional breakpoints.
    quadrature : {'gk21', 'gk15', 'trapezoid'}, optional
        Quadrature rule to use on subintervals.
        Options: 'gk21' (Gauss-Kronrod 21-point rule),
        'gk15' (Gauss-Kronrod 15-point rule),
        'trapezoid' (composite trapezoid rule).
        Default: 'gk21' for finite intervals and 'gk15' for (semi-)infinite
    full_output : bool, optional
        Return an additional ``info`` dictionary.
    args : tuple, optional
        Extra arguments to pass to function, if any.

        .. versionadded:: 1.8.0

    Returns
    -------
    res : {float, array-like}
        Estimate for the result
    err : float
        Error estimate for the result in the given norm
    info : dict
        Returned only when ``full_output=True``.
        Info dictionary. Is an object with the attributes:

            success : bool
                Whether integration reached target precision.
            status : int
                Indicator for convergence, success (0),
                failure (1), and failure due to rounding error (2).
            neval : int
                Number of function evaluations.
            intervals : ndarray, shape (num_intervals, 2)
                Start and end points of subdivision intervals.
            integrals : ndarray, shape (num_intervals, ...)
                Integral for each interval.
                Note that at most ``cache_size`` values are recorded,
                and the array may contains *nan* for missing items.
            errors : ndarray, shape (num_intervals,)
                Estimated integration error for each interval.

    Notes
    -----
    The algorithm mainly follows the implementation of QUADPACK's
    DQAG* algorithms, implementing global error control and adaptive
    subdivision.

    The algorithm here has some differences to the QUADPACK approach:

    Instead of subdividing one interval at a time, the algorithm
    subdivides N intervals with largest errors at once. This enables
    (partial) parallelization of the integration.

    The logic of subdividing "next largest" intervals first is then
    not implemented, and we rely on the above extension to avoid
    concentrating on "small" intervals only.

    The Wynn epsilon table extrapolation is not used (QUADPACK uses it
    for infinite intervals). This is because the algorithm here is
    supposed to work on vector-valued functions, in an user-specified
    norm, and the extension of the epsilon algorithm to this case does
    not appear to be widely agreed. For max-norm, using elementwise
    Wynn epsilon could be possible, but we do not do this here with
    the hope that the epsilon extrapolation is mainly useful in
    special cases.

    References
    ----------
    [1] R. Piessens, E. de Doncker, QUADPACK (1983).

    Examples
    --------
    We can compute integrations of a vector-valued function:

    >>> from scipy.integrate import quad_vec
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> alpha = np.linspace(0.0, 2.0, num=30)
    >>> f = lambda x: x**alpha
    >>> x0, x1 = 0, 2
    >>> y, err = quad_vec(f, x0, x1)
    >>> plt.plot(alpha, y)
    >>> plt.xlabel(r"$\\alpha$")
    >>> plt.ylabel(r"$\\int_{0}^{2} x^\\alpha dx$")
    >>> plt.show()

    """
    a = float(a)
    b = float(b)
    if args:
        if not isinstance(args, tuple):
            args = (args,)
        f = _FunctionWrapper(f, args)
    kwargs = dict(epsabs=epsabs, epsrel=epsrel, norm=norm, cache_size=cache_size, limit=limit, workers=workers, points=points, quadrature='gk15' if quadrature is None else quadrature, full_output=full_output)
    if np.isfinite(a) and np.isinf(b):
        f2 = SemiInfiniteFunc(f, start=a, infty=b)
        if points is not None:
            kwargs['points'] = tuple((f2.get_t(xp) for xp in points))
        return quad_vec(f2, 0, 1, **kwargs)
    elif np.isfinite(b) and np.isinf(a):
        f2 = SemiInfiniteFunc(f, start=b, infty=a)
        if points is not None:
            kwargs['points'] = tuple((f2.get_t(xp) for xp in points))
        res = quad_vec(f2, 0, 1, **kwargs)
        return (-res[0],) + res[1:]
    elif np.isinf(a) and np.isinf(b):
        sgn = -1 if b < a else 1
        f2 = DoubleInfiniteFunc(f)
        if points is not None:
            kwargs['points'] = (0,) + tuple((f2.get_t(xp) for xp in points))
        else:
            kwargs['points'] = (0,)
        if a != b:
            res = quad_vec(f2, -1, 1, **kwargs)
        else:
            res = quad_vec(f2, 1, 1, **kwargs)
        return (res[0] * sgn,) + res[1:]
    elif not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError(f'invalid integration bounds a={a}, b={b}')
    norm_funcs = {None: _max_norm, 'max': _max_norm, '2': np.linalg.norm}
    if callable(norm):
        norm_func = norm
    else:
        norm_func = norm_funcs[norm]
    parallel_count = 128
    min_intervals = 2
    try:
        _quadrature = {None: _quadrature_gk21, 'gk21': _quadrature_gk21, 'gk15': _quadrature_gk15, 'trapz': _quadrature_trapezoid, 'trapezoid': _quadrature_trapezoid}[quadrature]
    except KeyError as e:
        raise ValueError(f'unknown quadrature {quadrature!r}') from e
    if points is None:
        initial_intervals = [(a, b)]
    else:
        prev = a
        initial_intervals = []
        for p in sorted(points):
            p = float(p)
            if not a < p < b or p == prev:
                continue
            initial_intervals.append((prev, p))
            prev = p
        initial_intervals.append((prev, b))
    global_integral = None
    global_error = None
    rounding_error = None
    interval_cache = None
    intervals = []
    neval = 0
    for x1, x2 in initial_intervals:
        ig, err, rnd = _quadrature(x1, x2, f, norm_func)
        neval += _quadrature.num_eval
        if global_integral is None:
            if isinstance(ig, (float, complex)):
                if norm_func in (_max_norm, np.linalg.norm):
                    norm_func = abs
            global_integral = ig
            global_error = float(err)
            rounding_error = float(rnd)
            cache_count = cache_size // _get_sizeof(ig)
            interval_cache = LRUDict(cache_count)
        else:
            global_integral += ig
            global_error += err
            rounding_error += rnd
        interval_cache[x1, x2] = copy.copy(ig)
        intervals.append((-err, x1, x2))
    heapq.heapify(intervals)
    CONVERGED = 0
    NOT_CONVERGED = 1
    ROUNDING_ERROR = 2
    NOT_A_NUMBER = 3
    status_msg = {CONVERGED: 'Target precision reached.', NOT_CONVERGED: 'Target precision not reached.', ROUNDING_ERROR: 'Target precision could not be reached due to rounding error.', NOT_A_NUMBER: 'Non-finite values encountered.'}
    with MapWrapper(workers) as mapwrapper:
        ier = NOT_CONVERGED
        while intervals and len(intervals) < limit:
            tol = max(epsabs, epsrel * norm_func(global_integral))
            to_process = []
            err_sum = 0
            for j in range(parallel_count):
                if not intervals:
                    break
                if j > 0 and err_sum > global_error - tol / 8:
                    break
                interval = heapq.heappop(intervals)
                neg_old_err, a, b = interval
                old_int = interval_cache.pop((a, b), None)
                to_process.append(((-neg_old_err, a, b, old_int), f, norm_func, _quadrature))
                err_sum += -neg_old_err
            for parts in mapwrapper(_subdivide_interval, to_process):
                dint, derr, dround_err, subint, dneval = parts
                neval += dneval
                global_integral += dint
                global_error += derr
                rounding_error += dround_err
                for x in subint:
                    x1, x2, ig, err = x
                    interval_cache[x1, x2] = ig
                    heapq.heappush(intervals, (-err, x1, x2))
            if len(intervals) >= min_intervals:
                tol = max(epsabs, epsrel * norm_func(global_integral))
                if global_error < tol / 8:
                    ier = CONVERGED
                    break
                if global_error < rounding_error:
                    ier = ROUNDING_ERROR
                    break
            if not (np.isfinite(global_error) and np.isfinite(rounding_error)):
                ier = NOT_A_NUMBER
                break
    res = global_integral
    err = global_error + rounding_error
    if full_output:
        res_arr = np.asarray(res)
        dummy = np.full(res_arr.shape, np.nan, dtype=res_arr.dtype)
        integrals = np.array([interval_cache.get((z[1], z[2]), dummy) for z in intervals], dtype=res_arr.dtype)
        errors = np.array([-z[0] for z in intervals])
        intervals = np.array([[z[1], z[2]] for z in intervals])
        info = _Bunch(neval=neval, success=ier == CONVERGED, status=ier, message=status_msg[ier], intervals=intervals, integrals=integrals, errors=errors)
        return (res, err, info)
    else:
        return (res, err)
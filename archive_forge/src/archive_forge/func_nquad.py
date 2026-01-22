import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
def nquad(func, ranges, args=None, opts=None, full_output=False):
    """
    Integration over multiple variables.

    Wraps `quad` to enable integration over multiple variables.
    Various options allow improved integration of discontinuous functions, as
    well as the use of weighted integration, and generally finer control of the
    integration process.

    Parameters
    ----------
    func : {callable, scipy.LowLevelCallable}
        The function to be integrated. Has arguments of ``x0, ... xn``,
        ``t0, ... tm``, where integration is carried out over ``x0, ... xn``,
        which must be floats.  Where ``t0, ... tm`` are extra arguments
        passed in args.
        Function signature should be ``func(x0, x1, ..., xn, t0, t1, ..., tm)``.
        Integration is carried out in order.  That is, integration over ``x0``
        is the innermost integral, and ``xn`` is the outermost.

        If the user desires improved integration performance, then `f` may
        be a `scipy.LowLevelCallable` with one of the signatures::

            double func(int n, double *xx)
            double func(int n, double *xx, void *user_data)

        where ``n`` is the number of variables and args.  The ``xx`` array
        contains the coordinates and extra arguments. ``user_data`` is the data
        contained in the `scipy.LowLevelCallable`.
    ranges : iterable object
        Each element of ranges may be either a sequence  of 2 numbers, or else
        a callable that returns such a sequence. ``ranges[0]`` corresponds to
        integration over x0, and so on. If an element of ranges is a callable,
        then it will be called with all of the integration arguments available,
        as well as any parametric arguments. e.g., if
        ``func = f(x0, x1, x2, t0, t1)``, then ``ranges[0]`` may be defined as
        either ``(a, b)`` or else as ``(a, b) = range0(x1, x2, t0, t1)``.
    args : iterable object, optional
        Additional arguments ``t0, ... tn``, required by ``func``, ``ranges``,
        and ``opts``.
    opts : iterable object or dict, optional
        Options to be passed to `quad`. May be empty, a dict, or
        a sequence of dicts or functions that return a dict. If empty, the
        default options from scipy.integrate.quad are used. If a dict, the same
        options are used for all levels of integraion. If a sequence, then each
        element of the sequence corresponds to a particular integration. e.g.,
        ``opts[0]`` corresponds to integration over ``x0``, and so on. If a
        callable, the signature must be the same as for ``ranges``. The
        available options together with their default values are:

          - epsabs = 1.49e-08
          - epsrel = 1.49e-08
          - limit  = 50
          - points = None
          - weight = None
          - wvar   = None
          - wopts  = None

        For more information on these options, see `quad`.

    full_output : bool, optional
        Partial implementation of ``full_output`` from scipy.integrate.quad.
        The number of integrand function evaluations ``neval`` can be obtained
        by setting ``full_output=True`` when calling nquad.

    Returns
    -------
    result : float
        The result of the integration.
    abserr : float
        The maximum of the estimates of the absolute error in the various
        integration results.
    out_dict : dict, optional
        A dict containing additional information on the integration.

    See Also
    --------
    quad : 1-D numerical integration
    dblquad, tplquad : double and triple integrals
    fixed_quad : fixed-order Gaussian quadrature
    quadrature : adaptive Gaussian quadrature

    Notes
    -----
    For valid results, the integral must converge; behavior for divergent
    integrals is not guaranteed.

    **Details of QUADPACK level routines**

    `nquad` calls routines from the FORTRAN library QUADPACK. This section
    provides details on the conditions for each routine to be called and a
    short description of each routine. The routine called depends on
    `weight`, `points` and the integration limits `a` and `b`.

    ================  ==============  ==========  =====================
    QUADPACK routine  `weight`        `points`    infinite bounds
    ================  ==============  ==========  =====================
    qagse             None            No          No
    qagie             None            No          Yes
    qagpe             None            Yes         No
    qawoe             'sin', 'cos'    No          No
    qawfe             'sin', 'cos'    No          either `a` or `b`
    qawse             'alg*'          No          No
    qawce             'cauchy'        No          No
    ================  ==============  ==========  =====================

    The following provides a short description from [1]_ for each
    routine.

    qagse
        is an integrator based on globally adaptive interval
        subdivision in connection with extrapolation, which will
        eliminate the effects of integrand singularities of
        several types.
    qagie
        handles integration over infinite intervals. The infinite range is
        mapped onto a finite interval and subsequently the same strategy as
        in ``QAGS`` is applied.
    qagpe
        serves the same purposes as QAGS, but also allows the
        user to provide explicit information about the location
        and type of trouble-spots i.e. the abscissae of internal
        singularities, discontinuities and other difficulties of
        the integrand function.
    qawoe
        is an integrator for the evaluation of
        :math:`\\int^b_a \\cos(\\omega x)f(x)dx` or
        :math:`\\int^b_a \\sin(\\omega x)f(x)dx`
        over a finite interval [a,b], where :math:`\\omega` and :math:`f`
        are specified by the user. The rule evaluation component is based
        on the modified Clenshaw-Curtis technique

        An adaptive subdivision scheme is used in connection
        with an extrapolation procedure, which is a modification
        of that in ``QAGS`` and allows the algorithm to deal with
        singularities in :math:`f(x)`.
    qawfe
        calculates the Fourier transform
        :math:`\\int^\\infty_a \\cos(\\omega x)f(x)dx` or
        :math:`\\int^\\infty_a \\sin(\\omega x)f(x)dx`
        for user-provided :math:`\\omega` and :math:`f`. The procedure of
        ``QAWO`` is applied on successive finite intervals, and convergence
        acceleration by means of the :math:`\\varepsilon`-algorithm is applied
        to the series of integral approximations.
    qawse
        approximate :math:`\\int^b_a w(x)f(x)dx`, with :math:`a < b` where
        :math:`w(x) = (x-a)^{\\alpha}(b-x)^{\\beta}v(x)` with
        :math:`\\alpha,\\beta > -1`, where :math:`v(x)` may be one of the
        following functions: :math:`1`, :math:`\\log(x-a)`, :math:`\\log(b-x)`,
        :math:`\\log(x-a)\\log(b-x)`.

        The user specifies :math:`\\alpha`, :math:`\\beta` and the type of the
        function :math:`v`. A globally adaptive subdivision strategy is
        applied, with modified Clenshaw-Curtis integration on those
        subintervals which contain `a` or `b`.
    qawce
        compute :math:`\\int^b_a f(x) / (x-c)dx` where the integral must be
        interpreted as a Cauchy principal value integral, for user specified
        :math:`c` and :math:`f`. The strategy is globally adaptive. Modified
        Clenshaw-Curtis integration is used on those intervals containing the
        point :math:`x = c`.

    References
    ----------

    .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
           Überhuber, Christoph W.; Kahaner, David (1983).
           QUADPACK: A subroutine package for automatic integration.
           Springer-Verlag.
           ISBN 978-3-540-12553-2.

    Examples
    --------
    Calculate

    .. math::

        \\int^{1}_{-0.15} \\int^{0.8}_{0.13} \\int^{1}_{-1} \\int^{1}_{0}
        f(x_0, x_1, x_2, x_3) \\,dx_0 \\,dx_1 \\,dx_2 \\,dx_3 ,

    where

    .. math::

        f(x_0, x_1, x_2, x_3) = \\begin{cases}
          x_0^2+x_1 x_2-x_3^3+ \\sin{x_0}+1 & (x_0-0.2 x_3-0.5-0.25 x_1 > 0) \\\\
          x_0^2+x_1 x_2-x_3^3+ \\sin{x_0}+0 & (x_0-0.2 x_3-0.5-0.25 x_1 \\leq 0)
        \\end{cases} .

    >>> import numpy as np
    >>> from scipy import integrate
    >>> func = lambda x0,x1,x2,x3 : x0**2 + x1*x2 - x3**3 + np.sin(x0) + (
    ...                                 1 if (x0-.2*x3-.5-.25*x1>0) else 0)
    >>> def opts0(*args, **kwargs):
    ...     return {'points':[0.2*args[2] + 0.5 + 0.25*args[0]]}
    >>> integrate.nquad(func, [[0,1], [-1,1], [.13,.8], [-.15,1]],
    ...                 opts=[opts0,{},{},{}], full_output=True)
    (1.5267454070738633, 2.9437360001402324e-14, {'neval': 388962})

    Calculate

    .. math::

        \\int^{t_0+t_1+1}_{t_0+t_1-1}
        \\int^{x_2+t_0^2 t_1^3+1}_{x_2+t_0^2 t_1^3-1}
        \\int^{t_0 x_1+t_1 x_2+1}_{t_0 x_1+t_1 x_2-1}
        f(x_0,x_1, x_2,t_0,t_1)
        \\,dx_0 \\,dx_1 \\,dx_2,

    where

    .. math::

        f(x_0, x_1, x_2, t_0, t_1) = \\begin{cases}
          x_0 x_2^2 + \\sin{x_1}+2 & (x_0+t_1 x_1-t_0 > 0) \\\\
          x_0 x_2^2 +\\sin{x_1}+1 & (x_0+t_1 x_1-t_0 \\leq 0)
        \\end{cases}

    and :math:`(t_0, t_1) = (0, 1)` .

    >>> def func2(x0, x1, x2, t0, t1):
    ...     return x0*x2**2 + np.sin(x1) + 1 + (1 if x0+t1*x1-t0>0 else 0)
    >>> def lim0(x1, x2, t0, t1):
    ...     return [t0*x1 + t1*x2 - 1, t0*x1 + t1*x2 + 1]
    >>> def lim1(x2, t0, t1):
    ...     return [x2 + t0**2*t1**3 - 1, x2 + t0**2*t1**3 + 1]
    >>> def lim2(t0, t1):
    ...     return [t0 + t1 - 1, t0 + t1 + 1]
    >>> def opts0(x1, x2, t0, t1):
    ...     return {'points' : [t0 - t1*x1]}
    >>> def opts1(x2, t0, t1):
    ...     return {}
    >>> def opts2(t0, t1):
    ...     return {}
    >>> integrate.nquad(func2, [lim0, lim1, lim2], args=(0,1),
    ...                 opts=[opts0, opts1, opts2])
    (36.099919226771625, 1.8546948553373528e-07)

    """
    depth = len(ranges)
    ranges = [rng if callable(rng) else _RangeFunc(rng) for rng in ranges]
    if args is None:
        args = ()
    if opts is None:
        opts = [dict([])] * depth
    if isinstance(opts, dict):
        opts = [_OptFunc(opts)] * depth
    else:
        opts = [opt if callable(opt) else _OptFunc(opt) for opt in opts]
    return _NQuad(func, ranges, opts, full_output).integrate(*args)
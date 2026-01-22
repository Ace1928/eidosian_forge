import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
def dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08):
    """
    Compute a double integral.

    Return the double (definite) integral of ``func(y, x)`` from ``x = a..b``
    and ``y = gfun(x)..hfun(x)``.

    Parameters
    ----------
    func : callable
        A Python function or method of at least two variables: y must be the
        first argument and x the second argument.
    a, b : float
        The limits of integration in x: `a` < `b`
    gfun : callable or float
        The lower boundary curve in y which is a function taking a single
        floating point argument (x) and returning a floating point result
        or a float indicating a constant boundary curve.
    hfun : callable or float
        The upper boundary curve in y (same requirements as `gfun`).
    args : sequence, optional
        Extra arguments to pass to `func`.
    epsabs : float, optional
        Absolute tolerance passed directly to the inner 1-D quadrature
        integration. Default is 1.49e-8. ``dblquad`` tries to obtain
        an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
        where ``i`` = inner integral of ``func(y, x)`` from ``gfun(x)``
        to ``hfun(x)``, and ``result`` is the numerical approximation.
        See `epsrel` below.
    epsrel : float, optional
        Relative tolerance of the inner 1-D integrals. Default is 1.49e-8.
        If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
        and ``50 * (machine epsilon)``. See `epsabs` above.

    Returns
    -------
    y : float
        The resultant integral.
    abserr : float
        An estimate of the error.

    See Also
    --------
    quad : single integral
    tplquad : triple integral
    nquad : N-dimensional integrals
    fixed_quad : fixed-order Gaussian quadrature
    quadrature : adaptive Gaussian quadrature
    odeint : ODE integrator
    ode : ODE integrator
    simpson : integrator for sampled data
    romb : integrator for sampled data
    scipy.special : for coefficients and roots of orthogonal polynomials


    Notes
    -----
    For valid results, the integral must converge; behavior for divergent
    integrals is not guaranteed.

    **Details of QUADPACK level routines**

    `quad` calls routines from the FORTRAN library QUADPACK. This section
    provides details on the conditions for each routine to be called and a
    short description of each routine. For each level of integration, ``qagse``
    is used for finite limits or ``qagie`` is used if either limit (or both!)
    are infinite. The following provides a short description from [1]_ for each
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

    References
    ----------

    .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
           Überhuber, Christoph W.; Kahaner, David (1983).
           QUADPACK: A subroutine package for automatic integration.
           Springer-Verlag.
           ISBN 978-3-540-12553-2.

    Examples
    --------
    Compute the double integral of ``x * y**2`` over the box
    ``x`` ranging from 0 to 2 and ``y`` ranging from 0 to 1.
    That is, :math:`\\int^{x=2}_{x=0} \\int^{y=1}_{y=0} x y^2 \\,dy \\,dx`.

    >>> import numpy as np
    >>> from scipy import integrate
    >>> f = lambda y, x: x*y**2
    >>> integrate.dblquad(f, 0, 2, 0, 1)
        (0.6666666666666667, 7.401486830834377e-15)

    Calculate :math:`\\int^{x=\\pi/4}_{x=0} \\int^{y=\\cos(x)}_{y=\\sin(x)} 1
    \\,dy \\,dx`.

    >>> f = lambda y, x: 1
    >>> integrate.dblquad(f, 0, np.pi/4, np.sin, np.cos)
        (0.41421356237309503, 1.1083280054755938e-14)

    Calculate :math:`\\int^{x=1}_{x=0} \\int^{y=2-x}_{y=x} a x y \\,dy \\,dx`
    for :math:`a=1, 3`.

    >>> f = lambda y, x, a: a*x*y
    >>> integrate.dblquad(f, 0, 1, lambda x: x, lambda x: 2-x, args=(1,))
        (0.33333333333333337, 5.551115123125783e-15)
    >>> integrate.dblquad(f, 0, 1, lambda x: x, lambda x: 2-x, args=(3,))
        (0.9999999999999999, 1.6653345369377348e-14)

    Compute the two-dimensional Gaussian Integral, which is the integral of the
    Gaussian function :math:`f(x,y) = e^{-(x^{2} + y^{2})}`, over
    :math:`(-\\infty,+\\infty)`. That is, compute the integral
    :math:`\\iint^{+\\infty}_{-\\infty} e^{-(x^{2} + y^{2})} \\,dy\\,dx`.

    >>> f = lambda x, y: np.exp(-(x ** 2 + y ** 2))
    >>> integrate.dblquad(f, -np.inf, np.inf, -np.inf, np.inf)
        (3.141592653589777, 2.5173086737433208e-08)

    """

    def temp_ranges(*args):
        return [gfun(args[0]) if callable(gfun) else gfun, hfun(args[0]) if callable(hfun) else hfun]
    return nquad(func, [temp_ranges, [a, b]], args=args, opts={'epsabs': epsabs, 'epsrel': epsrel})
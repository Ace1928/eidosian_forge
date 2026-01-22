from ..libmp.backend import xrange
from .calculus import defun
@defun
def sumap(ctx, f, interval, integral=None, error=False):
    """
    Evaluates an infinite series of an analytic summand *f* using the
    Abel-Plana formula

    .. math ::

        \\sum_{k=0}^{\\infty} f(k) = \\int_0^{\\infty} f(t) dt + \\frac{1}{2} f(0) +
            i \\int_0^{\\infty} \\frac{f(it)-f(-it)}{e^{2\\pi t}-1} dt.

    Unlike the Euler-Maclaurin formula (see :func:`~mpmath.sumem`),
    the Abel-Plana formula does not require derivatives. However,
    it only works when `|f(it)-f(-it)|` does not
    increase too rapidly with `t`.

    **Examples**

    The Abel-Plana formula is particularly useful when the summand
    decreases like a power of `k`; for example when the sum is a pure
    zeta function::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> sumap(lambda k: 1/k**2.5, [1,inf])
        1.34148725725091717975677
        >>> zeta(2.5)
        1.34148725725091717975677
        >>> sumap(lambda k: 1/(k+1j)**(2.5+2.5j), [1,inf])
        (-3.385361068546473342286084 - 0.7432082105196321803869551j)
        >>> zeta(2.5+2.5j, 1+1j)
        (-3.385361068546473342286084 - 0.7432082105196321803869551j)

    If the series is alternating, numerical quadrature along the real
    line is likely to give poor results, so it is better to evaluate
    the first term symbolically whenever possible:

        >>> n=3; z=-0.75
        >>> I = expint(n,-log(z))
        >>> chop(sumap(lambda k: z**k / k**n, [1,inf], integral=I))
        -0.6917036036904594510141448
        >>> polylog(n,z)
        -0.6917036036904594510141448

    """
    prec = ctx.prec
    try:
        ctx.prec += 10
        a, b = interval
        if b != ctx.inf:
            raise ValueError('b should be equal to ctx.inf')
        g = lambda x: f(x + a)
        if integral is None:
            i1, err1 = ctx.quad(g, [0, ctx.inf], error=True)
        else:
            i1, err1 = (integral, 0)
        j = ctx.j
        p = ctx.pi * 2
        if ctx._is_real_type(i1):
            h = lambda t: -2 * ctx.im(g(j * t)) / ctx.expm1(p * t)
        else:
            h = lambda t: j * (g(j * t) - g(-j * t)) / ctx.expm1(p * t)
        i2, err2 = ctx.quad(h, [0, ctx.inf], error=True)
        err = err1 + err2
        v = i1 + i2 + 0.5 * g(ctx.mpf(0))
    finally:
        ctx.prec = prec
    if error:
        return (+v, err)
    return +v
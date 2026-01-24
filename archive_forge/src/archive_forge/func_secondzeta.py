from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun
def secondzeta(ctx, s, a=0.015, **kwargs):
    """
    Evaluates the secondary zeta function `Z(s)`, defined for
    `\\mathrm{Re}(s)>1` by

    .. math ::

        Z(s) = \\sum_{n=1}^{\\infty} \\frac{1}{\\tau_n^s}

    where `\\frac12+i\\tau_n` runs through the zeros of `\\zeta(s)` with
    imaginary part positive.

    `Z(s)` extends to a meromorphic function on `\\mathbb{C}`  with a
    double pole at `s=1` and  simple poles at the points `-2n` for
    `n=0`,  1, 2, ...

    **Examples**

        >>> from mpmath import *
        >>> mp.pretty = True; mp.dps = 15
        >>> secondzeta(2)
        0.023104993115419
        >>> xi = lambda s: 0.5*s*(s-1)*pi**(-0.5*s)*gamma(0.5*s)*zeta(s)
        >>> Xi = lambda t: xi(0.5+t*j)
        >>> chop(-0.5*diff(Xi,0,n=2)/Xi(0))
        0.023104993115419

    We may ask for an approximate error value::

        >>> secondzeta(0.5+100j, error=True)
        ((-0.216272011276718 - 0.844952708937228j), 2.22044604925031e-16)

    The function has poles at the negative odd integers,
    and dyadic rational values at the negative even integers::

        >>> mp.dps = 30
        >>> secondzeta(-8)
        -0.67236328125
        >>> secondzeta(-7)
        +inf

    **Implementation notes**

    The function is computed as sum of four terms `Z(s)=A(s)-P(s)+E(s)-S(s)`
    respectively main, prime, exponential and singular terms.
    The main term `A(s)` is computed from the zeros of zeta.
    The prime term depends on the von Mangoldt function.
    The singular term is responsible for the poles of the function.

    The four terms depends on a small parameter `a`. We may change the
    value of `a`. Theoretically this has no effect on the sum of the four
    terms, but in practice may be important.

    A smaller value of the parameter `a` makes `A(s)` depend on
    a smaller number of zeros of zeta, but `P(s)`  uses more values of
    von Mangoldt function.

    We may also add a verbose option to obtain data about the
    values of the four terms.

        >>> mp.dps = 10
        >>> secondzeta(0.5 + 40j, error=True, verbose=True)
        main term = (-30190318549.138656312556 - 13964804384.624622892823j)
            computed using 19 zeros of zeta
        prime term = (132717176.89212754625045 + 188980555.17563978290601j)
            computed using 9 values of the von Mangoldt function
        exponential term = (542447428666.07179812536 + 362434922978.80192435203j)
        singular term = (512124392939.98154322355 + 348281138038.65531023921j)
        ((0.059471043 + 0.3463514534j), 1.455191523e-11)

        >>> secondzeta(0.5 + 40j, a=0.04, error=True, verbose=True)
        main term = (-151962888.19606243907725 - 217930683.90210294051982j)
            computed using 9 zeros of zeta
        prime term = (2476659342.3038722372461 + 28711581821.921627163136j)
            computed using 37 values of the von Mangoldt function
        exponential term = (178506047114.7838188264 + 819674143244.45677330576j)
        singular term = (175877424884.22441310708 + 790744630738.28669174871j)
        ((0.059471043 + 0.3463514534j), 1.455191523e-11)

    Notice the great cancellation between the four terms. Changing `a`, the
    four terms are very different numbers but the cancellation gives
    the good value of Z(s).

    **References**

    A. Voros, Zeta functions for the Riemann zeros, Ann. Institute Fourier,
    53, (2003) 665--699.

    A. Voros, Zeta functions over Zeros of Zeta Functions, Lecture Notes
    of the Unione Matematica Italiana, Springer, 2009.
    """
    s = ctx.convert(s)
    a = ctx.convert(a)
    tol = ctx.eps
    if ctx.isint(s) and ctx.re(s) <= 1:
        if abs(s - 1) < tol * 1000:
            return ctx.inf
        m = int(round(ctx.re(s)))
        if m & 1:
            return ctx.inf
        else:
            return (-1) ** (-m // 2) * ctx.fraction(8 - ctx.eulernum(-m, exact=True), 2 ** (-m + 3))
    prec = ctx.prec
    try:
        t3 = secondzeta_exp_term(ctx, s, a)
        extraprec = max(ctx.mag(t3), 0)
        ctx.prec += extraprec + 3
        t1, r1, gt = secondzeta_main_term(ctx, s, a, error='True', verbose='True')
        t2, r2, pt = secondzeta_prime_term(ctx, s, a, error='True', verbose='True')
        t4, r4 = secondzeta_singular_term(ctx, s, a, error='True')
        t3 = secondzeta_exp_term(ctx, s, a)
        err = r1 + r2 + r4
        t = t1 - t2 + t3 - t4
        if kwargs.get('verbose'):
            print('main term =', t1)
            print('    computed using', gt, 'zeros of zeta')
            print('prime term =', t2)
            print('    computed using', pt, 'values of the von Mangoldt function')
            print('exponential term =', t3)
            print('singular term =', t4)
    finally:
        ctx.prec = prec
    if kwargs.get('error'):
        w = max(ctx.mag(abs(t)), 0)
        err = max(err * 2 ** w, ctx.eps * 1.0 * 2 ** w)
        return (+t, err)
    return +t
from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun_wrapped
def lerchphi(ctx, z, s, a):
    """
    Gives the Lerch transcendent, defined for `|z| < 1` and
    `\\Re{a} > 0` by

    .. math ::

        \\Phi(z,s,a) = \\sum_{k=0}^{\\infty} \\frac{z^k}{(a+k)^s}

    and generally by the recurrence `\\Phi(z,s,a) = z \\Phi(z,s,a+1) + a^{-s}`
    along with the integral representation valid for `\\Re{a} > 0`

    .. math ::

        \\Phi(z,s,a) = \\frac{1}{2 a^s} +
                \\int_0^{\\infty} \\frac{z^t}{(a+t)^s} dt -
                2 \\int_0^{\\infty} \\frac{\\sin(t \\log z - s
                    \\operatorname{arctan}(t/a)}{(a^2 + t^2)^{s/2}
                    (e^{2 \\pi t}-1)} dt.

    The Lerch transcendent generalizes the Hurwitz zeta function :func:`zeta`
    (`z = 1`) and the polylogarithm :func:`polylog` (`a = 1`).

    **Examples**

    Several evaluations in terms of simpler functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> lerchphi(-1,2,0.5); 4*catalan
        3.663862376708876060218414
        3.663862376708876060218414
        >>> diff(lerchphi, (-1,-2,1), (0,1,0)); 7*zeta(3)/(4*pi**2)
        0.2131391994087528954617607
        0.2131391994087528954617607
        >>> lerchphi(-4,1,1); log(5)/4
        0.4023594781085250936501898
        0.4023594781085250936501898
        >>> lerchphi(-3+2j,1,0.5); 2*atanh(sqrt(-3+2j))/sqrt(-3+2j)
        (1.142423447120257137774002 + 0.2118232380980201350495795j)
        (1.142423447120257137774002 + 0.2118232380980201350495795j)

    Evaluation works for complex arguments and `|z| \\ge 1`::

        >>> lerchphi(1+2j, 3-j, 4+2j)
        (0.002025009957009908600539469 + 0.003327897536813558807438089j)
        >>> lerchphi(-2,2,-2.5)
        -12.28676272353094275265944
        >>> lerchphi(10,10,10)
        (-4.462130727102185701817349e-11 - 1.575172198981096218823481e-12j)
        >>> lerchphi(10,10,-10.5)
        (112658784011940.5605789002 - 498113185.5756221777743631j)

    Some degenerate cases::

        >>> lerchphi(0,1,2)
        0.5
        >>> lerchphi(0,1,-2)
        -0.5

    Reduction to simpler functions::

        >>> lerchphi(1, 4.25+1j, 1)
        (1.044674457556746668033975 - 0.04674508654012658932271226j)
        >>> zeta(4.25+1j)
        (1.044674457556746668033975 - 0.04674508654012658932271226j)
        >>> lerchphi(1 - 0.5**10, 4.25+1j, 1)
        (1.044629338021507546737197 - 0.04667768813963388181708101j)
        >>> lerchphi(3, 4, 1)
        (1.249503297023366545192592 - 0.2314252413375664776474462j)
        >>> polylog(4, 3) / 3
        (1.249503297023366545192592 - 0.2314252413375664776474462j)
        >>> lerchphi(3, 4, 1 - 0.5**10)
        (1.253978063946663945672674 - 0.2316736622836535468928376j)

    **References**

    1. [DLMF]_ section 25.14

    """
    if z == 0:
        return a ** (-s)
    if z == 1:
        return ctx.zeta(s, a)
    if a == 1:
        return ctx.polylog(s, z) / z
    if ctx.re(a) < 1:
        if ctx.isnpint(a):
            raise ValueError('Lerch transcendent complex infinity')
        m = int(ctx.ceil(1 - ctx.re(a)))
        v = ctx.zero
        zpow = ctx.one
        for n in xrange(m):
            v += zpow / (a + n) ** s
            zpow *= z
        return zpow * ctx.lerchphi(z, s, a + m) + v
    g = ctx.ln(z)
    v = 1 / (2 * a ** s) + ctx.gammainc(1 - s, -a * g) * (-g) ** (s - 1) / z ** a
    h = s / 2
    r = 2 * ctx.pi
    f = lambda t: ctx.sin(s * ctx.atan(t / a) - t * g) / ((a ** 2 + t ** 2) ** h * ctx.expm1(r * t))
    v += 2 * ctx.quad(f, [0, ctx.inf])
    if not ctx.im(z) and (not ctx.im(s)) and (not ctx.im(a)) and (ctx.re(z) < 1):
        v = ctx.chop(v)
    return v
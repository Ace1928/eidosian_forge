from sympy.core.containers import Tuple
from sympy.core.function import Derivative
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (appellf1, hyper, meijerg)
from sympy.series.order import O
from sympy.abc import x, z, k
from sympy.series.limits import limit
from sympy.testing.pytest import raises, slow
from sympy.core.random import (
@slow
def test_hyperrep():
    from sympy.functions.special.hyper import HyperRep, HyperRep_atanh, HyperRep_power1, HyperRep_power2, HyperRep_log1, HyperRep_asin1, HyperRep_asin2, HyperRep_sqrts1, HyperRep_sqrts2, HyperRep_log2, HyperRep_cosasin, HyperRep_sinasin
    from sympy.functions.elementary.exponential import exp_polar
    from sympy.functions.elementary.piecewise import Piecewise
    a, b, c, d, z = symbols('a b c d z')

    class myrep(HyperRep):

        @classmethod
        def _expr_small(cls, x):
            return a

        @classmethod
        def _expr_small_minus(cls, x):
            return b

        @classmethod
        def _expr_big(cls, x, n):
            return c * n

        @classmethod
        def _expr_big_minus(cls, x, n):
            return d * n
    assert myrep(z).rewrite('nonrep') == Piecewise((0, abs(z) > 1), (a, True))
    assert myrep(exp_polar(I * pi) * z).rewrite('nonrep') == Piecewise((0, abs(z) > 1), (b, True))
    assert myrep(exp_polar(2 * I * pi) * z).rewrite('nonrep') == Piecewise((c, abs(z) > 1), (a, True))
    assert myrep(exp_polar(3 * I * pi) * z).rewrite('nonrep') == Piecewise((d, abs(z) > 1), (b, True))
    assert myrep(exp_polar(4 * I * pi) * z).rewrite('nonrep') == Piecewise((2 * c, abs(z) > 1), (a, True))
    assert myrep(exp_polar(5 * I * pi) * z).rewrite('nonrep') == Piecewise((2 * d, abs(z) > 1), (b, True))
    assert myrep(z).rewrite('nonrepsmall') == a
    assert myrep(exp_polar(I * pi) * z).rewrite('nonrepsmall') == b

    def t(func, hyp, z):
        """ Test that func is a valid representation of hyp. """
        if not tn(func.rewrite('nonrepsmall'), hyp, z, a=Rational(-1, 2), b=Rational(-1, 2), c=S.Half, d=S.Half):
            return False
        if not tn(func.rewrite('nonrepsmall').subs(z, exp_polar(I * pi) * z).replace(exp_polar, exp), func.subs(z, exp_polar(I * pi) * z).rewrite('nonrepsmall'), z, a=Rational(-1, 2), b=Rational(-1, 2), c=S.Half, d=S.Half):
            return False
        expr = func.subs(z, exp_polar(I * pi) * z).rewrite('nonrep')
        if abs(expr.subs(z, 1 + 1e-15).n() - expr.subs(z, 1 - 1e-15).n()) > 1e-10:
            return False

        def dosubs(func, a, b):
            rv = func.subs(z, exp_polar(a) * z).rewrite('nonrep')
            return rv.subs(z, exp_polar(b) * z).replace(exp_polar, exp)
        for n in [0, 1, 2, 3, 4, -1, -2, -3, -4]:
            expr1 = dosubs(func, 2 * I * pi * n, I * pi / 2)
            expr2 = dosubs(func, 2 * I * pi * n + I * pi, -I * pi / 2)
            if not tn(expr1, expr2, z):
                return False
            expr1 = dosubs(func, 2 * I * pi * (n + 1), -I * pi / 2)
            expr2 = dosubs(func, 2 * I * pi * n + I * pi, I * pi / 2)
            if not tn(expr1, expr2, z):
                return False
        return True
    a = Rational(1, 3)
    assert t(HyperRep_atanh(z), hyper([S.Half, 1], [Rational(3, 2)], z), z)
    assert t(HyperRep_power1(a, z), hyper([-a], [], z), z)
    assert t(HyperRep_power2(a, z), hyper([a, a - S.Half], [2 * a], z), z)
    assert t(HyperRep_log1(z), -z * hyper([1, 1], [2], z), z)
    assert t(HyperRep_asin1(z), hyper([S.Half, S.Half], [Rational(3, 2)], z), z)
    assert t(HyperRep_asin2(z), hyper([1, 1], [Rational(3, 2)], z), z)
    assert t(HyperRep_sqrts1(a, z), hyper([-a, S.Half - a], [S.Half], z), z)
    assert t(HyperRep_sqrts2(a, z), -2 * z / (2 * a + 1) * hyper([-a - S.Half, -a], [S.Half], z).diff(z), z)
    assert t(HyperRep_log2(z), -z / 4 * hyper([Rational(3, 2), 1, 1], [2, 2], z), z)
    assert t(HyperRep_cosasin(a, z), hyper([-a, a], [S.Half], z), z)
    assert t(HyperRep_sinasin(a, z), 2 * a * z * hyper([1 - a, 1 + a], [Rational(3, 2)], z), z)
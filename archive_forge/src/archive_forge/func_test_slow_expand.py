from itertools import product
from sympy.concrete.summations import Sum
from sympy.core.function import (diff, expand_func)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (conjugate, polar_lift)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.bessel import (besseli, besselj, besselk, bessely, hankel1, hankel2, hn1, hn2, jn, jn_zeros, yn)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.series.series import series
from sympy.functions.special.bessel import (airyai, airybi,
from sympy.core.random import (random_complex_number as randcplx,
from sympy.simplify import besselsimp
from sympy.testing.pytest import raises, slow
from sympy.abc import z, n, k, x
@slow
def test_slow_expand():

    def check(eq, ans):
        return tn(eq, ans) and eq == ans
    rn = randcplx(a=1, b=0, d=0, c=2)
    for besselx in [besselj, bessely, besseli, besselk]:
        ri = S(2 * randint(-11, 10) + 1) / 2
        assert tn(besselsimp(besselx(ri, z)), besselx(ri, z))
    assert check(expand_func(besseli(rn, x)), besseli(rn - 2, x) - 2 * (rn - 1) * besseli(rn - 1, x) / x)
    assert check(expand_func(besseli(-rn, x)), besseli(-rn + 2, x) + 2 * (-rn + 1) * besseli(-rn + 1, x) / x)
    assert check(expand_func(besselj(rn, x)), -besselj(rn - 2, x) + 2 * (rn - 1) * besselj(rn - 1, x) / x)
    assert check(expand_func(besselj(-rn, x)), -besselj(-rn + 2, x) + 2 * (-rn + 1) * besselj(-rn + 1, x) / x)
    assert check(expand_func(besselk(rn, x)), besselk(rn - 2, x) + 2 * (rn - 1) * besselk(rn - 1, x) / x)
    assert check(expand_func(besselk(-rn, x)), besselk(-rn + 2, x) - 2 * (-rn + 1) * besselk(-rn + 1, x) / x)
    assert check(expand_func(bessely(rn, x)), -bessely(rn - 2, x) + 2 * (rn - 1) * bessely(rn - 1, x) / x)
    assert check(expand_func(bessely(-rn, x)), -bessely(-rn + 2, x) + 2 * (-rn + 1) * bessely(-rn + 1, x) / x)
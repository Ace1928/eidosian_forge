from sympy.concrete.summations import Sum
from sympy.core.function import expand_func
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (Abs, polar_lift)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.zeta_functions import (dirichlet_eta, lerchphi, polylog, riemann_xi, stieltjes, zeta)
from sympy.series.order import O
from sympy.core.function import ArgumentIndexError
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.testing.pytest import raises
from sympy.core.random import (test_derivative_numerically as td,
def test_zeta_eval():
    assert zeta(nan) is nan
    assert zeta(x, nan) is nan
    assert zeta(0) == Rational(-1, 2)
    assert zeta(0, x) == S.Half - x
    assert zeta(0, b) == S.Half - b
    assert zeta(1) is zoo
    assert zeta(1, 2) is zoo
    assert zeta(1, -7) is zoo
    assert zeta(1, x) is zoo
    assert zeta(2, 1) == pi ** 2 / 6
    assert zeta(3, 1) == zeta(3)
    assert zeta(2) == pi ** 2 / 6
    assert zeta(4) == pi ** 4 / 90
    assert zeta(6) == pi ** 6 / 945
    assert zeta(4, 3) == pi ** 4 / 90 - Rational(17, 16)
    assert zeta(7, 4) == zeta(7) - Rational(282251, 279936)
    assert zeta(S.Half, 2).func == zeta
    assert expand_func(zeta(S.Half, 2)) == zeta(S.Half) - 1
    assert zeta(x, 3).func == zeta
    assert expand_func(zeta(x, 3)) == zeta(x) - 1 - 1 / 2 ** x
    assert zeta(2, 0) is nan
    assert zeta(3, -1) is nan
    assert zeta(4, -2) is nan
    assert zeta(oo) == 1
    assert zeta(-1) == Rational(-1, 12)
    assert zeta(-2) == 0
    assert zeta(-3) == Rational(1, 120)
    assert zeta(-4) == 0
    assert zeta(-5) == Rational(-1, 252)
    assert zeta(-1, 3) == Rational(-37, 12)
    assert zeta(-1, 7) == Rational(-253, 12)
    assert zeta(-1, -4) == Rational(-121, 12)
    assert zeta(-1, -9) == Rational(-541, 12)
    assert zeta(-4, 3) == -17
    assert zeta(-4, -8) == 8772
    assert zeta(0, 1) == Rational(-1, 2)
    assert zeta(0, -1) == Rational(3, 2)
    assert zeta(0, 2) == Rational(-3, 2)
    assert zeta(0, -2) == Rational(5, 2)
    assert zeta(3).evalf(20).epsilon_eq(Float('1.2020569031595942854', 20), 1e-19)
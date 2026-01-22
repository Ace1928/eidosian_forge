from sympy.core.random import randrange
from sympy.simplify.hyperexpand import (ShiftA, ShiftB, UnShiftA, UnShiftB,
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.abc import z, a, b, c
from sympy.testing.pytest import XFAIL, raises, slow, ON_CI, skip
from sympy.core.random import verify_numerically as tn
from sympy.core.numbers import (Rational, pi)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.functions.special.bessel import besseli
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
@XFAIL
def test_prudnikov_fail_other():
    assert can_do([1, a], [b, 1 - 2 * a + b])
    assert can_do([Rational(-1, 2)], [S.Half, 1])
    assert can_do([1], [S.Half, S.Half])
    assert can_do([Rational(1, 4)], [S.Half, Rational(5, 4)])
    assert can_do([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)])
    assert can_do([1], [Rational(1, 4), Rational(3, 4)])
    assert can_do([1], [Rational(3, 4), Rational(5, 4)])
    assert can_do([1], [Rational(5, 4), Rational(7, 4)])
    assert can_do([S.Half, 1], [Rational(3, 4), Rational(5, 4), Rational(3, 2)])
    assert can_do([S.Half, 1], [Rational(7, 4), Rational(5, 4), Rational(3, 2)])
    assert can_do([], [Rational(1, 3), S(2 / 3)])
    assert can_do([], [Rational(2, 3), S(4 / 3)])
    assert can_do([], [Rational(5, 3), S(4 / 3)])
    assert can_do([], [a, a + S.Half, 2 * a - 1])
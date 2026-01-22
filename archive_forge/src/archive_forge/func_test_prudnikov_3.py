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
@slow
def test_prudnikov_3():
    if ON_CI:
        skip('Too slow for CI.')
    h = S.Half
    assert can_do([Rational(1, 4), Rational(3, 4)], [h])
    assert can_do([Rational(1, 4), Rational(3, 4)], [3 * h])
    assert can_do([Rational(1, 3), Rational(2, 3)], [3 * h])
    assert can_do([Rational(3, 4), Rational(5, 4)], [h])
    assert can_do([Rational(3, 4), Rational(5, 4)], [3 * h])
    for p in [1, 2, 3, 4]:
        for n in [-h, h, 1, 3 * h, 2, 5 * h, 3, 7 * h, 4, 9 * h]:
            for m in [1, 3 * h, 2, 5 * h, 3, 7 * h, 4]:
                assert can_do([p, m], [n])
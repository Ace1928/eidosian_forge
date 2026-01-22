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
def test_prudnikov_8():
    h = S.Half
    for ai in [1, 2, 3]:
        for bi in [1, 2, 3]:
            for ci in range(1, ai + 1):
                for di in [h, 1, 3 * h, 2, 5 * h, 3]:
                    assert can_do([ai, bi], [ci, di])
        for bi in [3 * h, 5 * h]:
            for ci in [h, 1, 3 * h, 2, 5 * h, 3]:
                for di in [1, 2, 3]:
                    assert can_do([ai, bi], [ci, di])
    for ai in [-h, h, 3 * h, 5 * h]:
        for bi in [1, 2, 3]:
            for ci in [h, 1, 3 * h, 2, 5 * h, 3]:
                for di in [1, 2, 3]:
                    assert can_do([ai, bi], [ci, di])
        for bi in [h, 3 * h, 5 * h]:
            for ci in [h, 3 * h, 5 * h, 3]:
                for di in [h, 1, 3 * h, 2, 5 * h, 3]:
                    if ci <= bi:
                        assert can_do([ai, bi], [ci, di])
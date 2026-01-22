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
def test_meijerg_lookup():
    from sympy.functions.special.error_functions import Ci, Si
    from sympy.functions.special.gamma_functions import uppergamma
    assert hyperexpand(meijerg([a], [], [b, a], [], z)) == z ** b * exp(z) * gamma(-a + b + 1) * uppergamma(a - b, z)
    assert hyperexpand(meijerg([0], [], [0, 0], [], z)) == exp(z) * uppergamma(0, z)
    assert can_do_meijer([a], [], [b, a + 1], [])
    assert can_do_meijer([a], [], [b + 2, a], [])
    assert can_do_meijer([a], [], [b - 2, a], [])
    assert hyperexpand(meijerg([a], [], [a, a, a - S.Half], [], z)) == -sqrt(pi) * z ** (a - S.Half) * (2 * cos(2 * sqrt(z)) * (Si(2 * sqrt(z)) - pi / 2) - 2 * sin(2 * sqrt(z)) * Ci(2 * sqrt(z))) == hyperexpand(meijerg([a], [], [a, a - S.Half, a], [], z)) == hyperexpand(meijerg([a], [], [a - S.Half, a, a], [], z))
    assert can_do_meijer([a - 1], [], [a + 2, a - Rational(3, 2), a + 1], [])
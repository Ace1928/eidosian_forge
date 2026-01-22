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
def test_hyperexpand_special():
    assert hyperexpand(hyper([a, b], [c], 1)) == gamma(c) * gamma(c - a - b) / gamma(c - a) / gamma(c - b)
    assert hyperexpand(hyper([a, b], [1 + a - b], -1)) == gamma(1 + a / 2) * gamma(1 + a - b) / gamma(1 + a) / gamma(1 + a / 2 - b)
    assert hyperexpand(hyper([a, b], [1 + b - a], -1)) == gamma(1 + b / 2) * gamma(1 + b - a) / gamma(1 + b) / gamma(1 + b / 2 - a)
    assert hyperexpand(meijerg([1 - z - a / 2], [1 - z + a / 2], [b / 2], [-b / 2], 1)) == gamma(1 - 2 * z) * gamma(z + a / 2 + b / 2) / gamma(1 - z + a / 2 - b / 2) / gamma(1 - z - a / 2 + b / 2) / gamma(1 - z + a / 2 + b / 2)
    assert hyperexpand(hyper([a], [b], 0)) == 1
    assert hyper([a], [b], 0) != 0
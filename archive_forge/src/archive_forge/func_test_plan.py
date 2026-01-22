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
def test_plan():
    assert devise_plan(Hyper_Function([0], ()), Hyper_Function([0], ()), z) == []
    with raises(ValueError):
        devise_plan(Hyper_Function([1], ()), Hyper_Function((), ()), z)
    with raises(ValueError):
        devise_plan(Hyper_Function([2], [1]), Hyper_Function([2], [2]), z)
    with raises(ValueError):
        devise_plan(Hyper_Function([2], []), Hyper_Function([S('1/2')], []), z)
    a1, a2, b1 = (randcplx(n) for n in range(3))
    b1 += 2 * I
    h = hyper([a1, a2], [b1], z)
    h2 = hyper((a1 + 1, a2), [b1], z)
    assert tn(apply_operators(h, devise_plan(Hyper_Function((a1 + 1, a2), [b1]), Hyper_Function((a1, a2), [b1]), z), op), h2, z)
    h2 = hyper((a1 + 1, a2 - 1), [b1], z)
    assert tn(apply_operators(h, devise_plan(Hyper_Function((a1 + 1, a2 - 1), [b1]), Hyper_Function((a1, a2), [b1]), z), op), h2, z)
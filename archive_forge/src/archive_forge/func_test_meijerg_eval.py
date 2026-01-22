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
def test_meijerg_eval():
    from sympy.functions.elementary.exponential import exp_polar
    from sympy.functions.special.bessel import besseli
    from sympy.abc import l
    a = randcplx()
    arg = x * exp_polar(k * pi * I)
    expr1 = pi * meijerg([[], [(a + 1) / 2]], [[a / 2], [-a / 2, (a + 1) / 2]], arg ** 2 / 4)
    expr2 = besseli(a, arg)
    for x_ in [0.5, 1.5]:
        for k_ in [0.0, 0.1, 0.3, 0.5, 0.8, 1, 5.751, 15.3]:
            assert abs((expr1 - expr2).n(subs={x: x_, k: k_})) < 1e-10
            assert abs((expr1 - expr2).n(subs={x: x_, k: -k_})) < 1e-10
    eps = 1e-13
    expr2 = expr1.subs(k, l)
    for x_ in [0.5, 1.5]:
        for k_ in [0.5, Rational(1, 3), 0.25, 0.75, Rational(2, 3), 1.0, 1.5]:
            assert abs((expr1 - expr2).n(subs={x: x_, k: k_ + eps, l: k_ - eps})) < 1e-10
            assert abs((expr1 - expr2).n(subs={x: x_, k: -k_ + eps, l: -k_ - eps})) < 1e-10
    expr = (meijerg(((0.5,), ()), ((0.5, 0, 0.5), ()), exp_polar(-I * pi) / 4) + meijerg(((0.5,), ()), ((0.5, 0, 0.5), ()), exp_polar(I * pi) / 4)) / (2 * sqrt(pi))
    assert (expr - pi / exp(1)).n(chop=True) == 0
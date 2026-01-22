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
def test_hyper_unpolarify():
    from sympy.functions.elementary.exponential import exp_polar
    a = exp_polar(2 * pi * I) * x
    b = x
    assert hyper([], [], a).argument == b
    assert hyper([0], [], a).argument == a
    assert hyper([0], [0], a).argument == b
    assert hyper([0, 1], [0], a).argument == a
    assert hyper([0, 1], [0], exp_polar(2 * pi * I)).argument == 1
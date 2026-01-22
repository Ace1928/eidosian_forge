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
def test_polylog_expansion():
    assert polylog(s, 0) == 0
    assert polylog(s, 1) == zeta(s)
    assert polylog(s, -1) == -dirichlet_eta(s)
    assert polylog(s, exp_polar(I * pi * Rational(4, 3))) == polylog(s, exp(I * pi * Rational(4, 3)))
    assert polylog(s, exp_polar(I * pi) / 3) == polylog(s, exp(I * pi) / 3)
    assert myexpand(polylog(1, z), -log(1 - z))
    assert myexpand(polylog(0, z), z / (1 - z))
    assert myexpand(polylog(-1, z), z / (1 - z) ** 2)
    assert ((1 - z) ** 3 * expand_func(polylog(-2, z))).simplify() == z * (1 + z)
    assert myexpand(polylog(-5, z), None)
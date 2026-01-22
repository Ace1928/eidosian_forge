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
def test_dirichlet_eta_eval():
    assert dirichlet_eta(0) == S.Half
    assert dirichlet_eta(-1) == Rational(1, 4)
    assert dirichlet_eta(1) == log(2)
    assert dirichlet_eta(1, S.Half).simplify() == pi / 2
    assert dirichlet_eta(1, 2) == 1 - log(2)
    assert dirichlet_eta(2) == pi ** 2 / 12
    assert dirichlet_eta(4) == pi ** 4 * Rational(7, 720)
    assert str(dirichlet_eta(I).evalf(n=10)) == '0.5325931818 + 0.2293848577*I'
    assert str(dirichlet_eta(I, I).evalf(n=10)) == '3.462349253 + 0.220285771*I'
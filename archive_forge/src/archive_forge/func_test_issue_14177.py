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
def test_issue_14177():
    n = Symbol('n', nonnegative=True, integer=True)
    assert zeta(-n).rewrite(bernoulli) == bernoulli(n + 1) / (-n - 1)
    assert zeta(-n, a).rewrite(bernoulli) == bernoulli(n + 1, a) / (-n - 1)
    z2n = -(2 * I * pi) ** (2 * n) * bernoulli(2 * n) / (2 * factorial(2 * n))
    assert zeta(2 * n).rewrite(bernoulli) == z2n
    assert expand_func(zeta(s, n + 1)) == zeta(s) - harmonic(n, s)
    assert expand_func(zeta(-b, -n)) is nan
    assert expand_func(zeta(-b, n)) == zeta(-b, n)
    n = Symbol('n')
    assert zeta(2 * n) == zeta(2 * n)
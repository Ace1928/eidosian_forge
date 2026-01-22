from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (sech, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, atan, cos, sin, tan)
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import (Integral, integrate)
from sympy.testing.pytest import XFAIL, SKIP, slow, skip, ON_CI
from sympy.abc import x, k, c, y, b, h, a, m, z, n, t
@XFAIL
@slow
def test_issue_15227():
    if ON_CI:
        skip('Too slow for CI.')
    i = integrate(log(1 - x) * log((1 + x) ** 2) / x, (x, 0, 1))
    assert not i.has(Integral)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acot, atan, cos, sin)
from sympy.functions.elementary.complexes import sign as _sign
from sympy.functions.special.error_functions import (Ei, erf)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.polys.polytools import cancel
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh
from sympy.series.gruntz import compare, mrv, rewrite, mrv_leadterm, gruntz, \
from sympy.testing.pytest import XFAIL, skip, slow
def test_mrv3():
    assert mmrv(exp(x ** 2) + x * exp(x) + log(x) ** x / x, x) == {exp(x ** 2)}
    assert mmrv(exp(x) * (exp(1 / x + exp(-x)) - exp(1 / x)), x) == {exp(x), exp(-x)}
    assert mmrv(log(x ** 2 + 2 * exp(exp(3 * x ** 3 * log(x)))), x) == {exp(exp(3 * x ** 3 * log(x)))}
    assert mmrv(log(x - log(x)) / log(x), x) == {x}
    assert mmrv((exp(1 / x - exp(-x)) - exp(1 / x)) * exp(x), x) == {exp(x), exp(-x)}
    assert mmrv(1 / exp(-x + exp(-x)) - exp(x), x) == {exp(x), exp(-x), exp(x - exp(-x))}
    assert mmrv(log(log(x * exp(x * exp(x)) + 1)), x) == {exp(x * exp(x))}
    assert mmrv(exp(exp(log(log(x) + 1 / x))), x) == {x}
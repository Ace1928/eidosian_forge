from sympy.core.function import expand_func, Subs
from sympy.core import EulerGamma
from sympy.core.numbers import (I, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, atan)
from sympy.functions.special.error_functions import (Ei, erf, erfc)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma, lowergamma, multigamma, polygamma, trigamma, uppergamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
from sympy.core.random import (test_derivative_numerically as td,
def test_polygamma_expand_func():
    assert polygamma(0, x).expand(func=True) == polygamma(0, x)
    assert polygamma(0, 2 * x).expand(func=True) == polygamma(0, x) / 2 + polygamma(0, S.Half + x) / 2 + log(2)
    assert polygamma(1, 2 * x).expand(func=True) == polygamma(1, x) / 4 + polygamma(1, S.Half + x) / 4
    assert polygamma(2, x).expand(func=True) == polygamma(2, x)
    assert polygamma(0, -1 + x).expand(func=True) == polygamma(0, x) - 1 / (x - 1)
    assert polygamma(0, 1 + x).expand(func=True) == 1 / x + polygamma(0, x)
    assert polygamma(0, 2 + x).expand(func=True) == 1 / x + 1 / (1 + x) + polygamma(0, x)
    assert polygamma(0, 3 + x).expand(func=True) == polygamma(0, x) + 1 / x + 1 / (1 + x) + 1 / (2 + x)
    assert polygamma(0, 4 + x).expand(func=True) == polygamma(0, x) + 1 / x + 1 / (1 + x) + 1 / (2 + x) + 1 / (3 + x)
    assert polygamma(1, 1 + x).expand(func=True) == polygamma(1, x) - 1 / x ** 2
    assert polygamma(1, 2 + x).expand(func=True, multinomial=False) == polygamma(1, x) - 1 / x ** 2 - 1 / (1 + x) ** 2
    assert polygamma(1, 3 + x).expand(func=True, multinomial=False) == polygamma(1, x) - 1 / x ** 2 - 1 / (1 + x) ** 2 - 1 / (2 + x) ** 2
    assert polygamma(1, 4 + x).expand(func=True, multinomial=False) == polygamma(1, x) - 1 / x ** 2 - 1 / (1 + x) ** 2 - 1 / (2 + x) ** 2 - 1 / (3 + x) ** 2
    assert polygamma(0, x + y).expand(func=True) == polygamma(0, x + y)
    assert polygamma(1, x + y).expand(func=True) == polygamma(1, x + y)
    assert polygamma(1, 3 + 4 * x + y).expand(func=True, multinomial=False) == polygamma(1, y + 4 * x) - 1 / (y + 4 * x) ** 2 - 1 / (1 + y + 4 * x) ** 2 - 1 / (2 + y + 4 * x) ** 2
    assert polygamma(3, 3 + 4 * x + y).expand(func=True, multinomial=False) == polygamma(3, y + 4 * x) - 6 / (y + 4 * x) ** 4 - 6 / (1 + y + 4 * x) ** 4 - 6 / (2 + y + 4 * x) ** 4
    assert polygamma(3, 4 * x + y + 1).expand(func=True, multinomial=False) == polygamma(3, y + 4 * x) - 6 / (y + 4 * x) ** 4
    e = polygamma(3, 4 * x + y + Rational(3, 2))
    assert e.expand(func=True) == e
    e = polygamma(3, x + y + Rational(3, 4))
    assert e.expand(func=True, basic=False) == e
    assert polygamma(-1, x, evaluate=False).expand(func=True) == loggamma(x) - log(pi) / 2 - log(2) / 2
    p2 = polygamma(-2, x).expand(func=True) + x ** 2 / 2 - x / 2 + S(1) / 12
    assert isinstance(p2, Subs)
    assert p2.point == (-1,)
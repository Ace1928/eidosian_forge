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
def test_digamma_expand_func():
    assert digamma(x).expand(func=True) == polygamma(0, x)
    assert digamma(2 * x).expand(func=True) == polygamma(0, x) / 2 + polygamma(0, Rational(1, 2) + x) / 2 + log(2)
    assert digamma(-1 + x).expand(func=True) == polygamma(0, x) - 1 / (x - 1)
    assert digamma(1 + x).expand(func=True) == 1 / x + polygamma(0, x)
    assert digamma(2 + x).expand(func=True) == 1 / x + 1 / (1 + x) + polygamma(0, x)
    assert digamma(3 + x).expand(func=True) == polygamma(0, x) + 1 / x + 1 / (1 + x) + 1 / (2 + x)
    assert digamma(4 + x).expand(func=True) == polygamma(0, x) + 1 / x + 1 / (1 + x) + 1 / (2 + x) + 1 / (3 + x)
    assert digamma(x + y).expand(func=True) == polygamma(0, x + y)
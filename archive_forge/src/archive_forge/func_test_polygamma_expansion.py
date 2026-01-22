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
def test_polygamma_expansion():
    assert polygamma(0, 1 / x).nseries(x, n=3) == -log(x) - x / 2 - x ** 2 / 12 + O(x ** 3)
    assert polygamma(1, 1 / x).series(x, n=5) == x + x ** 2 / 2 + x ** 3 / 6 + O(x ** 5)
    assert polygamma(3, 1 / x).nseries(x, n=11) == 2 * x ** 3 + 3 * x ** 4 + 2 * x ** 5 - x ** 7 + 4 * x ** 9 / 3 + O(x ** 11)
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
def test_trigamma():
    assert trigamma(nan) == nan
    assert trigamma(oo) == 0
    assert trigamma(1) == pi ** 2 / 6
    assert trigamma(2) == pi ** 2 / 6 - 1
    assert trigamma(3) == pi ** 2 / 6 - Rational(5, 4)
    assert trigamma(x, evaluate=False).rewrite(zeta) == zeta(2, x)
    assert trigamma(x, evaluate=False).rewrite(harmonic) == trigamma(x).rewrite(polygamma).rewrite(harmonic)
    assert trigamma(x, evaluate=False).fdiff() == polygamma(2, x)
    assert trigamma(x, evaluate=False).is_real is None
    assert trigamma(x, evaluate=False).is_positive is None
    assert trigamma(x, evaluate=False).is_negative is None
    assert trigamma(x, evaluate=False).rewrite(polygamma) == polygamma(1, x)
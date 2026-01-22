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
def test_lowergamma():
    from sympy.functions.special.error_functions import expint
    from sympy.functions.special.hyper import meijerg
    assert lowergamma(x, 0) == 0
    assert lowergamma(x, y).diff(y) == y ** (x - 1) * exp(-y)
    assert td(lowergamma(randcplx(), y), y)
    assert td(lowergamma(x, randcplx()), x)
    assert lowergamma(x, y).diff(x) == gamma(x) * digamma(x) - uppergamma(x, y) * log(y) - meijerg([], [1, 1], [0, 0, x], [], y)
    assert lowergamma(S.Half, x) == sqrt(pi) * erf(sqrt(x))
    assert not lowergamma(S.Half - 3, x).has(lowergamma)
    assert not lowergamma(S.Half + 3, x).has(lowergamma)
    assert lowergamma(S.Half, x, evaluate=False).has(lowergamma)
    assert tn(lowergamma(S.Half + 3, x, evaluate=False), lowergamma(S.Half + 3, x), x)
    assert tn(lowergamma(S.Half - 3, x, evaluate=False), lowergamma(S.Half - 3, x), x)
    assert tn_branch(-3, lowergamma)
    assert tn_branch(-4, lowergamma)
    assert tn_branch(Rational(1, 3), lowergamma)
    assert tn_branch(pi, lowergamma)
    assert lowergamma(3, exp_polar(4 * pi * I) * x) == lowergamma(3, x)
    assert lowergamma(y, exp_polar(5 * pi * I) * x) == exp(4 * I * pi * y) * lowergamma(y, x * exp_polar(pi * I))
    assert lowergamma(-2, exp_polar(5 * pi * I) * x) == lowergamma(-2, x * exp_polar(I * pi)) + 2 * pi * I
    assert conjugate(lowergamma(x, y)) == lowergamma(conjugate(x), conjugate(y))
    assert conjugate(lowergamma(x, 0)) == 0
    assert unchanged(conjugate, lowergamma(x, -oo))
    assert lowergamma(0, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(S(1) / 3, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1, x, evaluate=False)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(x, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(x + 1, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1 / x, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(0, x + 1)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(S(1) / 3, x + 1)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(1, x + 1, evaluate=False)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(x, x + 1)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(x + 1, x + 1)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(1 / x, x + 1)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(0, 1 / x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(S(1) / 3, 1 / x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1, 1 / x, evaluate=False)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(x, 1 / x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(x + 1, 1 / x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1 / x, 1 / x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(x, 2).series(x, oo, 3) == 2 ** x * (1 + 2 / (x + 1)) * exp(-2) / x + O(exp(x * log(2)) / x ** 3, (x, oo))
    assert lowergamma(x, y).rewrite(expint) == -y ** x * expint(-x + 1, y) + gamma(x)
    k = Symbol('k', integer=True)
    assert lowergamma(k, y).rewrite(expint) == -y ** k * expint(-k + 1, y) + gamma(k)
    k = Symbol('k', integer=True, positive=False)
    assert lowergamma(k, y).rewrite(expint) == lowergamma(k, y)
    assert lowergamma(x, y).rewrite(uppergamma) == gamma(x) - uppergamma(x, y)
    assert lowergamma(70, 6) == factorial(69) - 69035724522603011058660187038367026272747334489677105069435923032634389419656200387949342530805432320 * exp(-6)
    assert (lowergamma(S(77) / 2, 6) - lowergamma(S(77) / 2, 6, evaluate=False)).evalf() < 1e-16
    assert (lowergamma(-S(77) / 2, 6) - lowergamma(-S(77) / 2, 6, evaluate=False)).evalf() < 1e-16
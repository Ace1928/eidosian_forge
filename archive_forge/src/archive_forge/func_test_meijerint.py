from sympy.core.function import expand_func
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.complexes import Abs, arg, re, unpolarify
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import cosh, acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, asin)
from sympy.functions.special.error_functions import (erf, erfc)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import simplify
from sympy.integrals.meijerint import (_rewrite_single, _rewrite1,
from sympy.testing.pytest import slow
from sympy.core.random import (verify_numerically,
from sympy.abc import x, y, a, b, c, d, s, t, z
@slow
def test_meijerint():
    from sympy.core.function import expand
    from sympy.core.symbol import symbols
    s, t, mu = symbols('s t mu', real=True)
    assert integrate(meijerg([], [], [0], [], s * t) * meijerg([], [], [mu / 2], [-mu / 2], t ** 2 / 4), (t, 0, oo)).is_Piecewise
    s = symbols('s', positive=True)
    assert integrate(x ** s * meijerg([[], []], [[0], []], x), (x, 0, oo)) == gamma(s + 1)
    assert integrate(x ** s * meijerg([[], []], [[0], []], x), (x, 0, oo), meijerg=True) == gamma(s + 1)
    assert isinstance(integrate(x ** s * meijerg([[], []], [[0], []], x), (x, 0, oo), meijerg=False), Integral)
    assert meijerint_indefinite(exp(x), x) == exp(x)
    a, b = symbols('a b', positive=True)
    assert simplify(meijerint_definite(x ** a, x, 0, b)[0]) == b ** (a + 1) / (a + 1)
    assert meijerint_definite((x + 1) ** 3 * exp(-x), x, 0, oo) == (16, True)
    sigma, mu = symbols('sigma mu', positive=True)
    i, c = meijerint_definite(exp(-((x - mu) / (2 * sigma)) ** 2), x, 0, oo)
    assert simplify(i) == sqrt(pi) * sigma * (2 - erfc(mu / (2 * sigma)))
    assert c == True
    i, _ = meijerint_definite(exp(-mu * x) * exp(sigma * x), x, 0, oo)
    assert simplify(i) == 1 / (mu - sigma)
    assert meijerint_definite(exp(x), x, -oo, 2) == (exp(2), True)
    assert expand(meijerint_definite(exp(x), x, 0, I)[0]) == exp(I) - 1
    assert expand(meijerint_definite(exp(-x), x, 0, x)[0]) == 1 - exp(-exp(I * arg(x)) * abs(x))
    assert meijerint_definite(exp(-x ** 2), x, -oo, oo) == (sqrt(pi), True)
    assert meijerint_definite(exp(-abs(x)), x, -oo, oo) == (2, True)
    assert meijerint_definite(exp(-(2 * x - 3) ** 2), x, -oo, oo) == (sqrt(pi) / 2, True)
    assert meijerint_definite(exp(-abs(2 * x - 3)), x, -oo, oo) == (1, True)
    assert meijerint_definite(exp(-((x - mu) / sigma) ** 2 / 2) / sqrt(2 * pi * sigma ** 2), x, -oo, oo) == (1, True)
    assert meijerint_definite(sinc(x) ** 2, x, -oo, oo) == (pi, True)
    assert meijerint_definite(exp(-x) * sin(x), x, 0, oo) == (S.Half, True)

    def res(n):
        return (1 / (1 + x ** 2)).diff(x, n).subs(x, 1) * (-1) ** n
    for n in range(6):
        assert integrate(exp(-x) * sin(x) * x ** n, (x, 0, oo), meijerg=True) == res(n)
    assert simplify(integrate(exp(-x) * sin(x + a), (x, 0, oo), meijerg=True)) == sqrt(2) * sin(a + pi / 4) / 2
    a, b, s = symbols('a b s')
    assert meijerint_definite(meijerg([], [], [a / 2], [-a / 2], x / 4) * meijerg([], [], [b / 2], [-b / 2], x / 4) * x ** (s - 1), x, 0, oo) == (4 * 2 ** (2 * s - 2) * gamma(-2 * s + 1) * gamma(a / 2 + b / 2 + s) / (gamma(-a / 2 + b / 2 - s + 1) * gamma(a / 2 - b / 2 - s + 1) * gamma(a / 2 + b / 2 - s + 1)), (re(s) < 1) & (re(s) < S(1) / 2) & (re(a) / 2 + re(b) / 2 + re(s) > 0))
    assert integrate(sin(x ** a) * sin(x ** b), (x, 0, oo), meijerg=True) == Integral(sin(x ** a) * sin(x ** b), (x, 0, oo))
    assert integrate(exp(-x ** 2) * log(x), (x, 0, oo), meijerg=True) == (sqrt(pi) * polygamma(0, S.Half) / 4).expand()
    from sympy.functions.special.gamma_functions import lowergamma
    n = symbols('n', integer=True)
    assert simplify(integrate(exp(-x) * x ** n, x, meijerg=True)) == lowergamma(n + 1, x)
    alpha = symbols('alpha', positive=True)
    assert meijerint_definite((2 - x) ** alpha * sin(alpha / x), x, 0, 2) == (sqrt(pi) * alpha * gamma(alpha + 1) * meijerg(((), (alpha / 2 + S.Half, alpha / 2 + 1)), ((0, 0, S.Half), (Rational(-1, 2),)), alpha ** 2 / 16) / 4, True)
    a, s = symbols('a s', positive=True)
    assert simplify(integrate(x ** s * exp(-a * x ** 2), (x, -oo, oo))) == a ** (-s / 2 - S.Half) * ((-1) ** s + 1) * gamma(s / 2 + S.Half) / 2
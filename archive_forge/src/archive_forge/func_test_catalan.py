import string
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import (diff, expand_func)
from sympy.core import (EulerGamma, TribonacciConstant)
from sympy.core.numbers import (Float, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.numbers import carmichael
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.integers import floor
from sympy.polys.polytools import cancel
from sympy.series.limits import limit, Limit
from sympy.series.order import O
from sympy.functions import (
from sympy.functions.combinatorial.numbers import _nT
from sympy.core.expr import unchanged
from sympy.core.numbers import GoldenRatio, Integer
from sympy.testing.pytest import raises, nocache_fail, warns_deprecated_sympy
from sympy.abc import x
def test_catalan():
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True, positive=True)
    k = Symbol('k', integer=True, nonnegative=True)
    p = Symbol('p', nonnegative=True)
    catalans = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786]
    for i, c in enumerate(catalans):
        assert catalan(i) == c
        assert catalan(n).rewrite(factorial).subs(n, i) == c
        assert catalan(n).rewrite(Product).subs(n, i).doit() == c
    assert unchanged(catalan, x)
    assert catalan(2 * x).rewrite(binomial) == binomial(4 * x, 2 * x) / (2 * x + 1)
    assert catalan(S.Half).rewrite(gamma) == 8 / (3 * pi)
    assert catalan(S.Half).rewrite(factorial).rewrite(gamma) == 8 / (3 * pi)
    assert catalan(3 * x).rewrite(gamma) == 4 ** (3 * x) * gamma(3 * x + S.Half) / (sqrt(pi) * gamma(3 * x + 2))
    assert catalan(x).rewrite(hyper) == hyper((-x + 1, -x), (2,), 1)
    assert catalan(n).rewrite(factorial) == factorial(2 * n) / (factorial(n + 1) * factorial(n))
    assert isinstance(catalan(n).rewrite(Product), catalan)
    assert isinstance(catalan(m).rewrite(Product), Product)
    assert diff(catalan(x), x) == (polygamma(0, x + S.Half) - polygamma(0, x + 2) + log(4)) * catalan(x)
    assert catalan(x).evalf() == catalan(x)
    c = catalan(S.Half).evalf()
    assert str(c) == '0.848826363156775'
    c = catalan(I).evalf(3)
    assert str((re(c), im(c))) == '(0.398, -0.0209)'
    assert catalan(p).is_positive is True
    assert catalan(k).is_integer is True
    assert catalan(m + 3).is_composite is True
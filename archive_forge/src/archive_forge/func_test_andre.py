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
def test_andre():
    nums = [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]
    for n, a in enumerate(nums):
        assert andre(n) == a
    assert andre(S.Infinity) == S.Infinity
    assert andre(-1) == -log(2)
    assert andre(-2) == -2 * S.Catalan
    assert andre(-3) == 3 * zeta(3) / 16
    assert andre(-5) == -15 * zeta(5) / 256
    assert unchanged(andre, -4)
    n = Symbol('n', integer=True, nonnegative=True)
    assert unchanged(andre, n)
    assert andre(n).is_integer is True
    assert andre(n).is_positive is True
    assert str(andre(10, evaluate=False).evalf(n=10)) == '50521.00000'
    assert str(andre(-1, evaluate=False).evalf(n=10)) == '-0.6931471806'
    assert str(andre(-2, evaluate=False).evalf(n=10)) == '-1.831931188'
    assert str(andre(-4, evaluate=False).evalf(n=10)) == '1.977889103'
    assert str(andre(I, evaluate=False).evalf(n=10)) == '2.378417833 + 0.6343322845*I'
    assert andre(x).rewrite(polylog) == (-I) ** (x + 1) * polylog(-x, I) + I ** (x + 1) * polylog(-x, -I)
    assert andre(x).rewrite(zeta) == 2 * gamma(x + 1) / (2 * pi) ** (x + 1) * (zeta(x + 1, Rational(1, 4)) - cos(pi * x) * zeta(x + 1, Rational(3, 4)))
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
def test_genocchi():
    genocchis = [0, -1, -1, 0, 1, 0, -3, 0, 17]
    for n, g in enumerate(genocchis):
        assert genocchi(n) == g
    m = Symbol('m', integer=True)
    n = Symbol('n', integer=True, positive=True)
    assert unchanged(genocchi, m)
    assert genocchi(2 * n + 1) == 0
    gn = 2 * (1 - 2 ** n) * bernoulli(n)
    assert genocchi(n).rewrite(bernoulli).factor() == gn.factor()
    gnx = 2 * (bernoulli(n, x) - 2 ** n * bernoulli(n, (x + 1) / 2))
    assert genocchi(n, x).rewrite(bernoulli).factor() == gnx.factor()
    assert genocchi(2 * n).is_odd
    assert genocchi(2 * n).is_even is False
    assert genocchi(2 * n + 1).is_even
    assert genocchi(n).is_integer
    assert genocchi(4 * n).is_positive
    assert genocchi(6, evaluate=False).is_prime == S(-3).is_prime
    assert genocchi(8, evaluate=False).is_prime
    assert genocchi(4 * n + 2).is_negative
    assert genocchi(4 * n + 1).is_negative is False
    assert genocchi(4 * n - 2).is_negative
    g0 = genocchi(0, evaluate=False)
    assert g0.is_positive is False
    assert g0.is_negative is False
    assert g0.is_even is True
    assert g0.is_odd is False
    assert genocchi(0, x) == 0
    assert genocchi(1, x) == -1
    assert genocchi(2, x) == 1 - 2 * x
    assert genocchi(3, x) == 3 * x - 3 * x ** 2
    assert genocchi(4, x) == -1 + 6 * x ** 2 - 4 * x ** 3
    y = Symbol('y')
    assert genocchi(5, (x + y) ** 100) == -5 * (x + y) ** 400 + 10 * (x + y) ** 300 - 5 * (x + y) ** 100
    assert str(genocchi(5.0, 4.0).evalf(n=10)) == '-660.0000000'
    assert str(genocchi(Rational(5, 4)).evalf(n=10)) == '-1.104286457'
    assert str(genocchi(-2).evalf(n=10)) == '3.606170709'
    assert str(genocchi(1.3, 3.7).evalf(n=10)) == '-1.847375373'
    assert str(genocchi(I, 1.0).evalf(n=10)) == '-0.3161917278 - 1.45311955*I'
    n = Symbol('n')
    assert genocchi(n, x).rewrite(dirichlet_eta) == -2 * n * dirichlet_eta(1 - n, x)
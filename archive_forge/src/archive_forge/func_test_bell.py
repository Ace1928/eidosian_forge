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
@nocache_fail
def test_bell():
    assert [bell(n) for n in range(8)] == [1, 1, 2, 5, 15, 52, 203, 877]
    assert bell(0, x) == 1
    assert bell(1, x) == x
    assert bell(2, x) == x ** 2 + x
    assert bell(5, x) == x ** 5 + 10 * x ** 4 + 25 * x ** 3 + 15 * x ** 2 + x
    assert bell(oo) is S.Infinity
    raises(ValueError, lambda: bell(oo, x))
    raises(ValueError, lambda: bell(-1))
    raises(ValueError, lambda: bell(S.Half))
    X = symbols('x:6')
    assert bell(6, 2, X[1:]) == 6 * X[5] * X[1] + 15 * X[4] * X[2] + 10 * X[3] ** 2
    assert bell(6, 3, X[1:]) == 15 * X[4] * X[1] ** 2 + 60 * X[3] * X[2] * X[1] + 15 * X[2] ** 3
    X = (1, 10, 100, 1000, 10000)
    assert bell(6, 2, X) == (6 + 15 + 10) * 10000
    X = (1, 2, 3, 3, 5)
    assert bell(6, 2, X) == 6 * 5 + 15 * 3 * 2 + 10 * 3 ** 2
    X = (1, 2, 3, 5)
    assert bell(6, 3, X) == 15 * 5 + 60 * 3 * 2 + 15 * 2 ** 3
    n = Symbol('n', integer=True, nonnegative=True)
    for i in [0, 2, 3, 7, 13, 42, 55]:
        assert bell(i).evalf() == bell(n).rewrite(Sum).evalf(subs={n: i})
    m = Symbol('m')
    assert bell(m).rewrite(Sum) == bell(m)
    assert bell(n, m).rewrite(Sum) == bell(n, m)
    n = Dummy('n')
    assert bell(n).limit(n, S.Infinity) is S.Infinity
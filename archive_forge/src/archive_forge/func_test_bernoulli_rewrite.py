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
def test_bernoulli_rewrite():
    from sympy.functions.elementary.piecewise import Piecewise
    n = Symbol('n', integer=True, nonnegative=True)
    assert bernoulli(-1).rewrite(zeta) == pi ** 2 / 6
    assert bernoulli(-2).rewrite(zeta) == 2 * zeta(3)
    assert not bernoulli(n, -3).rewrite(zeta).has(harmonic)
    assert bernoulli(-4, x).rewrite(zeta) == 4 * zeta(5, x)
    assert isinstance(bernoulli(n, x).rewrite(zeta), Piecewise)
    assert bernoulli(n + 1, x).rewrite(zeta) == -(n + 1) * zeta(-n, x)
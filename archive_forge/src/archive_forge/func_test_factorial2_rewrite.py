from sympy.concrete.products import Product
from sympy.core.function import expand_func
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core import EulerGamma
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (ff, rf, binomial, factorial, factorial2)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.functions.combinatorial.factorials import subfactorial
from sympy.functions.special.gamma_functions import uppergamma
from sympy.testing.pytest import XFAIL, raises, slow
def test_factorial2_rewrite():
    n = Symbol('n', integer=True)
    assert factorial2(n).rewrite(gamma) == 2 ** (n / 2) * Piecewise((1, Eq(Mod(n, 2), 0)), (sqrt(2) / sqrt(pi), Eq(Mod(n, 2), 1))) * gamma(n / 2 + 1)
    assert factorial2(2 * n).rewrite(gamma) == 2 ** n * gamma(n + 1)
    assert factorial2(2 * n + 1).rewrite(gamma) == sqrt(2) * 2 ** (n + S.Half) * gamma(n + Rational(3, 2)) / sqrt(pi)
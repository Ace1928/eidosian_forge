from sympy.core.function import (Function, Lambda, expand)
from sympy.core.numbers import (I, Rational)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import factor
from sympy.solvers.recurr import rsolve, rsolve_hyper, rsolve_poly, rsolve_ratio
from sympy.testing.pytest import raises, slow, XFAIL
from sympy.abc import a, b
def test_diofantissue_294():
    f = y(n) - y(n - 1) - 2 * y(n - 2) - 2 * n
    assert rsolve(f, y(n)) == (-1) ** n * C0 + 2 ** n * C1 - n - Rational(5, 2)
    assert rsolve(f, y(n), {y(0): -1, y(1): 1}) == -(-1) ** n / 2 + 2 * 2 ** n - n - Rational(5, 2)
    assert rsolve(-2 * y(n) + y(n + 1) + n - 1, y(n)) == 2 ** n * C0 + n
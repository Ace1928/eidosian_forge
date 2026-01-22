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
def test_issue_15553():
    f = Function('f')
    assert rsolve(Eq(f(n), 2 * f(n - 1) + n), f(n)) == 2 ** n * C0 - n - 2
    assert rsolve(Eq(f(n + 1), 2 * f(n) + n ** 2 + 1), f(n)) == 2 ** n * C0 - n ** 2 - 2 * n - 4
    assert rsolve(Eq(f(n + 1), 2 * f(n) + n ** 2 + 1), f(n), {f(1): 0}) == 7 * 2 ** n / 2 - n ** 2 - 2 * n - 4
    assert rsolve(Eq(f(n), 2 * f(n - 1) + 3 * n ** 2), f(n)) == 2 ** n * C0 - 3 * n ** 2 - 12 * n - 18
    assert rsolve(Eq(f(n), 2 * f(n - 1) + n ** 2), f(n)) == 2 ** n * C0 - n ** 2 - 4 * n - 6
    assert rsolve(Eq(f(n), 2 * f(n - 1) + n), f(n), {f(0): 1}) == 3 * 2 ** n - n - 2
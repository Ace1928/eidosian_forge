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
def test_issue_6844():
    f = y(n + 2) - y(n + 1) + y(n) / 4
    assert rsolve(f, y(n)) == 2 ** (-n + 1) * C1 * n + 2 ** (-n) * C0
    assert rsolve(f, y(n), {y(0): 0, y(1): 1}) == 2 ** (1 - n) * n
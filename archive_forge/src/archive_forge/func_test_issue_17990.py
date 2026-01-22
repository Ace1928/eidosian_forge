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
def test_issue_17990():
    f = -10 * y(n) + 4 * y(n + 1) + 6 * y(n + 2) + 46 * y(n + 3)
    sol = rsolve(f, y(n))
    expected = C0 * ((86 * 18 ** (S(1) / 3) / 69 + (-12 + (-1 + sqrt(3) * I) * (290412 + 3036 * sqrt(9165)) ** (S(1) / 3)) * (1 - sqrt(3) * I) * (24201 + 253 * sqrt(9165)) ** (S(1) / 3) / 276) / ((1 - sqrt(3) * I) * (24201 + 253 * sqrt(9165)) ** (S(1) / 3))) ** n + C1 * ((86 * 18 ** (S(1) / 3) / 69 + (-12 + (-1 - sqrt(3) * I) * (290412 + 3036 * sqrt(9165)) ** (S(1) / 3)) * (1 + sqrt(3) * I) * (24201 + 253 * sqrt(9165)) ** (S(1) / 3) / 276) / ((1 + sqrt(3) * I) * (24201 + 253 * sqrt(9165)) ** (S(1) / 3))) ** n + C2 * (-43 * 18 ** (S(1) / 3) / (69 * (24201 + 253 * sqrt(9165)) ** (S(1) / 3)) - S(1) / 23 + (290412 + 3036 * sqrt(9165)) ** (S(1) / 3) / 138) ** n
    assert sol == expected
    e = sol.subs({C0: 1, C1: 1, C2: 1, n: 1}).evalf()
    assert abs(e + 0.130434782608696) < 1e-13
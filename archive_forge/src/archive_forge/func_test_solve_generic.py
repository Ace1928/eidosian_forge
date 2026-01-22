from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polyerrors import UnsolvableFactorError
from sympy.polys.polyoptions import Options
from sympy.polys.polytools import Poly
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import flatten
from sympy.abc import x, y, z
from sympy.polys import PolynomialError
from sympy.solvers.polysys import (solve_poly_system,
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.testing.pytest import raises
def test_solve_generic():
    NewOption = Options((x, y), {'domain': 'ZZ'})
    assert solve_generic([x ** 2 - 2 * y ** 2, y ** 2 - y + 1], NewOption) == [(-sqrt(-1 - sqrt(3) * I), Rational(1, 2) - sqrt(3) * I / 2), (sqrt(-1 - sqrt(3) * I), Rational(1, 2) - sqrt(3) * I / 2), (-sqrt(-1 + sqrt(3) * I), Rational(1, 2) + sqrt(3) * I / 2), (sqrt(-1 + sqrt(3) * I), Rational(1, 2) + sqrt(3) * I / 2)]
    assert solve_generic([2 * x - y, (y - 1) * (y ** 5 - y + 1)], NewOption, strict=False) == [(Rational(1, 2), 1)]
    raises(UnsolvableFactorError, lambda: solve_generic([2 * x - y, (y - 1) * (y ** 5 - y + 1)], NewOption, strict=True))
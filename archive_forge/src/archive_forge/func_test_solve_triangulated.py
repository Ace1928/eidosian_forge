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
def test_solve_triangulated():
    f_1 = x ** 2 + y + z - 1
    f_2 = x + y ** 2 + z - 1
    f_3 = x + y + z ** 2 - 1
    a, b = (sqrt(2) - 1, -sqrt(2) - 1)
    assert solve_triangulated([f_1, f_2, f_3], x, y, z) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    dom = QQ.algebraic_field(sqrt(2))
    assert solve_triangulated([f_1, f_2, f_3], x, y, z, domain=dom) == [(0, 0, 1), (0, 1, 0), (1, 0, 0), (a, a, a), (b, b, b)]
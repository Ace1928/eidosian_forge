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
def test_solve_poly_system():
    assert solve_poly_system([x - 1], x) == [(S.One,)]
    assert solve_poly_system([y - x, y - x - 1], x, y) is None
    assert solve_poly_system([y - x ** 2, y + x ** 2], x, y) == [(S.Zero, S.Zero)]
    assert solve_poly_system([2 * x - 3, y * Rational(3, 2) - 2 * x, z - 5 * y], x, y, z) == [(Rational(3, 2), Integer(2), Integer(10))]
    assert solve_poly_system([x * y - 2 * y, 2 * y ** 2 - x ** 2], x, y) == [(0, 0), (2, -sqrt(2)), (2, sqrt(2))]
    assert solve_poly_system([y - x ** 2, y + x ** 2 + 1], x, y) == [(-I * sqrt(S.Half), Rational(-1, 2)), (I * sqrt(S.Half), Rational(-1, 2))]
    f_1 = x ** 2 + y + z - 1
    f_2 = x + y ** 2 + z - 1
    f_3 = x + y + z ** 2 - 1
    a, b = (sqrt(2) - 1, -sqrt(2) - 1)
    assert solve_poly_system([f_1, f_2, f_3], x, y, z) == [(0, 0, 1), (0, 1, 0), (1, 0, 0), (a, a, a), (b, b, b)]
    solution = [(1, -1), (1, 1)]
    assert solve_poly_system([Poly(x ** 2 - y ** 2), Poly(x - 1)]) == solution
    assert solve_poly_system([x ** 2 - y ** 2, x - 1], x, y) == solution
    assert solve_poly_system([x ** 2 - y ** 2, x - 1]) == solution
    assert solve_poly_system([x + x * y - 3, y + x * y - 4], x, y) == [(-3, -2), (1, 2)]
    raises(NotImplementedError, lambda: solve_poly_system([x ** 3 - y ** 3], x, y))
    raises(NotImplementedError, lambda: solve_poly_system([z, -2 * x * y ** 2 + x + y ** 2 * z, y ** 2 * (-z - 4) + 2]))
    raises(PolynomialError, lambda: solve_poly_system([1 / x], x))
    raises(NotImplementedError, lambda: solve_poly_system([x - 1], (x, y)))
    raises(NotImplementedError, lambda: solve_poly_system([y - 1], (x, y)))
    assert solve_poly_system([x ** 5 - x + 1], [x], strict=False) == []
    raises(UnsolvableFactorError, lambda: solve_poly_system([x ** 5 - x + 1], [x], strict=True))
    assert solve_poly_system([(x - 1) * (x ** 5 - x + 1), y ** 2 - 1], [x, y], strict=False) == [(1, -1), (1, 1)]
    raises(UnsolvableFactorError, lambda: solve_poly_system([(x - 1) * (x ** 5 - x + 1), y ** 2 - 1], [x, y], strict=True))
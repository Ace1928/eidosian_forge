from math import isclose
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda, nfloat, diff)
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, oo, pi, Integer)
from sympy.core.relational import (Eq, Gt, Ne, Ge)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign, conjugate)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
from sympy.functions.special.error_functions import (erf, erfc,
from sympy.logic.boolalg import And
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.sets.contains import Contains
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import ImageSet, Range
from sympy.sets.sets import (Complement, FiniteSet,
from sympy.simplify import simplify
from sympy.tensor.indexed import Indexed
from sympy.utilities.iterables import numbered_symbols
from sympy.testing.pytest import (XFAIL, raises, skip, slow, SKIP, _both_exp_pow)
from sympy.core.random import verify_numerically as tn
from sympy.physics.units import cm
from sympy.solvers import solve
from sympy.solvers.solveset import (
from sympy.abc import (a, b, c, d, e, f, g, h, i, j, k, l, m, n, q, r,
def test_linear_eq_to_matrix():
    assert linear_eq_to_matrix(0, x) == (Matrix([[0]]), Matrix([[0]]))
    assert linear_eq_to_matrix(1, x) == (Matrix([[0]]), Matrix([[-1]]))
    eqns1 = [2 * x + y - 2 * z - 3, x - y - z, x + y + 3 * z - 12]
    eqns2 = [Eq(3 * x + 2 * y - z, 1), Eq(2 * x - 2 * y + 4 * z, -2), -2 * x + y - 2 * z]
    A, B = linear_eq_to_matrix(eqns1, x, y, z)
    assert A == Matrix([[2, 1, -2], [1, -1, -1], [1, 1, 3]])
    assert B == Matrix([[3], [0], [12]])
    A, B = linear_eq_to_matrix(eqns2, x, y, z)
    assert A == Matrix([[3, 2, -1], [2, -2, 4], [-2, 1, -2]])
    assert B == Matrix([[1], [-2], [0]])
    eqns3 = [a * b * x + b * y + c * z - d, e * x + d * x + f * y + g * z - h, i * x + j * y + k * z - l]
    A, B = linear_eq_to_matrix(eqns3, x, y, z)
    assert A == Matrix([[a * b, b, c], [d + e, f, g], [i, j, k]])
    assert B == Matrix([[d], [h], [l]])
    raises(ValueError, lambda: linear_eq_to_matrix(eqns3))
    raises(ValueError, lambda: linear_eq_to_matrix(eqns3, [x, x, y]))
    raises(NonlinearError, lambda: linear_eq_to_matrix(Eq(1 / x + x, 1 / x), [x]))
    raises(NonlinearError, lambda: linear_eq_to_matrix([x ** 2], [x]))
    raises(NonlinearError, lambda: linear_eq_to_matrix([x * y], [x, y]))
    raises(ValueError, lambda: linear_eq_to_matrix(Eq(x, x), x))
    raises(ValueError, lambda: linear_eq_to_matrix(Eq(x, x + 1), x))
    assert linear_eq_to_matrix([x], [1 / x]) == (Matrix([[0]]), Matrix([[-x]]))
    assert linear_eq_to_matrix(x + y * (z * (3 * x + 2) + 3), x) == (Matrix([[3 * y * z + 1]]), Matrix([[-y * (2 * z + 3)]]))
    assert linear_eq_to_matrix(Matrix([[a * x + b * y - 7], [5 * x + 6 * y - c]]), x, y) == (Matrix([[a, b], [5, 6]]), Matrix([[7], [c]]))
    assert linear_eq_to_matrix(Eq(x + 2, 1), x) == (Matrix([[1]]), Matrix([[-1]]))
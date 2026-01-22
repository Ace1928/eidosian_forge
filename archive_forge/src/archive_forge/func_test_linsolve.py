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
def test_linsolve():
    x1, x2, x3, x4 = symbols('x1, x2, x3, x4')
    M = Matrix([[1, 2, 1, 1, 7], [1, 2, 2, -1, 12], [2, 4, 0, 6, 4]])
    system1 = A, B = (M[:, :-1], M[:, -1])
    Eqns = [x1 + 2 * x2 + x3 + x4 - 7, x1 + 2 * x2 + 2 * x3 - x4 - 12, 2 * x1 + 4 * x2 + 6 * x4 - 4]
    sol = FiniteSet((-2 * x2 - 3 * x4 + 2, x2, 2 * x4 + 5, x4))
    assert linsolve(Eqns, (x1, x2, x3, x4)) == sol
    assert linsolve(Eqns, *(x1, x2, x3, x4)) == sol
    assert linsolve(system1, (x1, x2, x3, x4)) == sol
    assert linsolve(system1, *(x1, x2, x3, x4)) == sol
    x1, x2, x3, x4 = symbols('x:4', cls=Dummy)
    assert linsolve(system1, x1, x2, x3, x4) == FiniteSet((-2 * x2 - 3 * x4 + 2, x2, 2 * x4 + 5, x4))
    raises(ValueError, lambda: linsolve(Eqns))
    raises(ValueError, lambda: linsolve(x1))
    raises(ValueError, lambda: linsolve(x1, x2))
    raises(ValueError, lambda: linsolve((A,), x1, x2))
    raises(ValueError, lambda: linsolve(A, B, x1, x2))
    raises(ValueError, lambda: linsolve([x1], x1, x1))
    raises(ValueError, lambda: linsolve([x1], (i for i in (x1, x1))))
    raises(NonlinearError, lambda: linsolve([x + y - 1, x ** 2 + y - 3], [x, y]))
    raises(NonlinearError, lambda: linsolve([cos(x) + y, x + y], [x, y]))
    assert linsolve([x + z - 1, x ** 2 + y - 3], [z, y]) == {(-x + 1, -x ** 2 + 3)}
    A = Matrix([[a, b], [c, d]])
    B = Matrix([[e], [g]])
    system2 = (A, B)
    sol = FiniteSet(((-b * g + d * e) / (a * d - b * c), (a * g - c * e) / (a * d - b * c)))
    assert linsolve(system2, [x, y]) == sol
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    B = Matrix([0, 0, 1])
    assert linsolve((A, B), (x, y, z)) is S.EmptySet
    A, B, J1, J2 = symbols('A B J1 J2')
    Augmatrix = Matrix([[2 * I * J1, 2 * I * J2, -2 / J1], [-2 * I * J2, -2 * I * J1, 2 / J2], [0, 2, 2 * I / (J1 * J2)], [2, 0, 0]])
    assert linsolve(Augmatrix, A, B) == FiniteSet((0, I / (J1 * J2)))
    Augmatrix = Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    assert linsolve(Augmatrix, a, b, c, d, e) == FiniteSet((a, 0, c, 0, e))
    x0, x1, x2, _x0 = symbols('tau0 tau1 tau2 _tau0')
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])) == FiniteSet((x0, 0, x1, _x0, x2))
    x0, x1, x2, _x0 = symbols('tau00 tau01 tau02 tau0')
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])) == FiniteSet((x0, 0, x1, _x0, x2))
    x0, x1, x2, _x0 = symbols('tau00 tau01 tau02 tau1')
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])) == FiniteSet((x0, 0, x1, _x0, x2))
    x0, x2, x4 = symbols('x0, x2, x4')
    assert linsolve(Augmatrix, numbered_symbols('x')) == FiniteSet((x0, 0, x2, 0, x4))
    Augmatrix[-1, -1] = x0
    Augmatrix[-1, -1] = symbols('_x0')
    assert len(linsolve(Augmatrix, numbered_symbols('x', cls=Dummy)).free_symbols) == 4
    f = Function('f')
    assert linsolve([f(x) - 5], f(x)) == FiniteSet((5,))
    from sympy.physics.units import meter, newton, kilo
    kN = kilo * newton
    Eqns = [8 * kN + x + y, 28 * kN * meter + 3 * x * meter]
    assert linsolve(Eqns, x, y) == {(kilo * newton * Rational(-28, 3), kN * Rational(4, 3))}
    assert linsolve([Eq(x, x + y)], [x, y]) == {(x, 0)}
    assert linsolve([Eq(x + x * y, 1 + y)], [x]) == {(1,)}
    assert linsolve([Eq(1 + y, x + x * y)], [x]) == {(1,)}
    raises(NonlinearError, lambda: linsolve([Eq(x ** 2, x ** 2 + y)], [x, y]))
    assert linsolve([], [x]) is S.EmptySet
    assert linsolve([0], [x]) == {(x,)}
    assert linsolve([x], [x, y]) == {(0, y)}
    assert linsolve([x, 0], [x, y]) == {(0, y)}
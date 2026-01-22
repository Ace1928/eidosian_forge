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
def test_solveset_sqrt_2():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert solveset_real(sqrt(2 * x - 1) - sqrt(x - 4) - 2, x) == FiniteSet(S(5), S(13))
    assert solveset_real(sqrt(x + 7) + 2 - sqrt(3 - x), x) == FiniteSet(-6)
    assert solveset_real(sqrt(17 * x - sqrt(x ** 2 - 5)) - 7, x) == FiniteSet(3)
    eq = x + 1 - (x ** 4 + 4 * x ** 3 - x) ** Rational(1, 4)
    assert solveset_real(eq, x) == FiniteSet(Rational(-1, 2), Rational(-1, 3))
    eq = sqrt(2 * x + 9) - sqrt(x + 1) - sqrt(x + 4)
    assert solveset_real(eq, x) == FiniteSet(0)
    eq = sqrt(x + 4) + sqrt(2 * x - 1) - 3 * sqrt(x - 1)
    assert solveset_real(eq, x) == FiniteSet(5)
    eq = sqrt(x) * sqrt(x - 7) - 12
    assert solveset_real(eq, x) == FiniteSet(16)
    eq = sqrt(x - 3) + sqrt(x) - 3
    assert solveset_real(eq, x) == FiniteSet(4)
    eq = sqrt(2 * x ** 2 - 7) - (3 - x)
    assert solveset_real(eq, x) == FiniteSet(-S(8), S(2))
    eq = sqrt(9 * x ** 2 + 4) - (3 * x + 2)
    assert solveset_real(eq, x) == FiniteSet(0)
    assert solveset_real(sqrt(x - 3) - sqrt(x) - 3, x) == FiniteSet()
    eq = (2 * x - 5) ** Rational(1, 3) - 3
    assert solveset_real(eq, x) == FiniteSet(16)
    assert solveset_real(sqrt(x) + sqrt(sqrt(x)) - 4, x) == FiniteSet((Rational(-1, 2) + sqrt(17) / 2) ** 4)
    eq = sqrt(x) - sqrt(x - 1) + sqrt(sqrt(x))
    assert solveset_real(eq, x) == FiniteSet()
    eq = (x - 4) ** 2 + (sqrt(x) - 2) ** 4
    assert solveset_real(eq, x) == FiniteSet(-4, 4)
    eq = sqrt(x) + sqrt(x + 1) + sqrt(1 - x) - 6 * sqrt(5) / 5
    ans = solveset_real(eq, x)
    ra = S('-1484/375 - 4*(-S(1)/2 + sqrt(3)*I/2)*(-12459439/52734375 +\n    114*sqrt(12657)/78125)**(S(1)/3) - 172564/(140625*(-S(1)/2 +\n    sqrt(3)*I/2)*(-12459439/52734375 + 114*sqrt(12657)/78125)**(S(1)/3))')
    rb = Rational(4, 5)
    assert all((abs(eq.subs(x, i).n()) < 1e-10 for i in (ra, rb))) and len(ans) == 2 and ({i.n(chop=True) for i in ans} == {i.n(chop=True) for i in (ra, rb)})
    assert solveset_real(sqrt(x) + x ** Rational(1, 3) + x ** Rational(1, 4), x) == FiniteSet(0)
    assert solveset_real(x / sqrt(x ** 2 + 1), x) == FiniteSet(0)
    eq = (x - y ** 3) / (y ** 2 * sqrt(1 - y ** 2))
    assert solveset_real(eq, x) == FiniteSet(y ** 3)
    assert solveset_real(1 / (5 + x) ** Rational(1, 5) - 9, x) == FiniteSet(Rational(-295244, 59049))
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
def test__solveset_multi():
    from sympy.solvers.solveset import _solveset_multi
    from sympy.sets import Reals
    assert _solveset_multi([x ** 2 - 1], [x], [S.Reals]) == FiniteSet((1,), (-1,))
    assert _solveset_multi([x + y, x + 1], [x, y], [Reals, Reals]) == FiniteSet((-1, 1))
    assert _solveset_multi([x + y, x + 1], [y, x], [Reals, Reals]) == FiniteSet((1, -1))
    assert _solveset_multi([x + y, x - y - 1], [x, y], [Reals, Reals]) == FiniteSet((S(1) / 2, -S(1) / 2))
    assert _solveset_multi([x - 1, y - 2], [x, y], [Reals, Reals]) == FiniteSet((1, 2))
    assert dumeq(_solveset_multi([x + y], [x, y], [Reals, Reals]), Union(ImageSet(Lambda(((x,),), (x, -x)), ProductSet(Reals)), ImageSet(Lambda(((y,),), (-y, y)), ProductSet(Reals))))
    assert _solveset_multi([x + y, x + y + 1], [x, y], [Reals, Reals]) == S.EmptySet
    assert _solveset_multi([x + y, x - y, x - 1], [x, y], [Reals, Reals]) == S.EmptySet
    assert _solveset_multi([x + y, x - y, x - 1], [y, x], [Reals, Reals]) == S.EmptySet
    assert _solveset_multi([x + y + z - 1, x + y - z - 2, x - y - z - 3], [x, y, z], [Reals, Reals, Reals]) == FiniteSet((2, -S.Half, -S.Half))
    from sympy.abc import theta
    assert _solveset_multi([x ** 2 + y ** 2 - 2, x + y], [x, y], [Reals, Reals]) == FiniteSet((-1, 1), (1, -1))
    assert _solveset_multi([x ** 2 - 1, y], [x, y], [Reals, Reals]) == FiniteSet((1, 0), (-1, 0))
    assert dumeq(_solveset_multi([x ** 2 - y ** 2], [x, y], [Reals, Reals]), Union(ImageSet(Lambda(((x,),), (x, -Abs(x))), ProductSet(Reals)), ImageSet(Lambda(((x,),), (x, Abs(x))), ProductSet(Reals)), ImageSet(Lambda(((y,),), (-Abs(y), y)), ProductSet(Reals)), ImageSet(Lambda(((y,),), (Abs(y), y)), ProductSet(Reals))))
    assert _solveset_multi([r * cos(theta) - 1, r * sin(theta)], [theta, r], [Interval(0, pi), Interval(-1, 1)]) == FiniteSet((0, 1), (pi, -1))
    assert _solveset_multi([r * cos(theta) - 1, r * sin(theta)], [r, theta], [Interval(0, 1), Interval(0, pi)]) == FiniteSet((1, 0))
    assert dumeq(_solveset_multi([r * cos(theta) - r, r * sin(theta)], [r, theta], [Interval(0, 1), Interval(0, pi)]), Union(ImageSet(Lambda(((r,),), (r, 0)), ImageSet(Lambda(r, (r,)), Interval(0, 1))), ImageSet(Lambda(((theta,),), (0, theta)), ImageSet(Lambda(theta, (theta,)), Interval(0, pi)))))
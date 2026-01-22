from sympy.core.expr import unchanged
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range, normalize_theta_set,
from sympy.sets.sets import (FiniteSet, Interval, Union, imageset,
from sympy.sets.conditionset import ConditionSet
from sympy.simplify.simplify import simplify
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import And
from sympy.matrices.dense import eye
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, y, t, z
from sympy.core.mod import Mod
import itertools
def test_imageset_intersect_interval():
    from sympy.abc import n
    f1 = ImageSet(Lambda(n, n * pi), S.Integers)
    f2 = ImageSet(Lambda(n, 2 * n), Interval(0, pi))
    f3 = ImageSet(Lambda(n, 2 * n * pi + pi / 2), S.Integers)
    f4 = ImageSet(Lambda(n, n * I * pi), S.Integers)
    f5 = ImageSet(Lambda(n, 2 * I * n * pi + pi / 2), S.Integers)
    f6 = ImageSet(Lambda(n, log(n)), S.Integers)
    f7 = ImageSet(Lambda(n, n ** 2), S.Integers)
    f8 = ImageSet(Lambda(n, Abs(n)), S.Integers)
    f9 = ImageSet(Lambda(n, exp(n)), S.Naturals0)
    assert f1.intersect(Interval(-1, 1)) == FiniteSet(0)
    assert f1.intersect(Interval(0, 2 * pi, False, True)) == FiniteSet(0, pi)
    assert f2.intersect(Interval(1, 2)) == Interval(1, 2)
    assert f3.intersect(Interval(-1, 1)) == S.EmptySet
    assert f3.intersect(Interval(-5, 5)) == FiniteSet(pi * Rational(-3, 2), pi / 2)
    assert f4.intersect(Interval(-1, 1)) == FiniteSet(0)
    assert f4.intersect(Interval(1, 2)) == S.EmptySet
    assert f5.intersect(Interval(0, 1)) == S.EmptySet
    assert f6.intersect(Interval(0, 1)) == FiniteSet(S.Zero, log(2))
    assert f7.intersect(Interval(0, 10)) == Intersection(f7, Interval(0, 10))
    assert f8.intersect(Interval(0, 2)) == Intersection(f8, Interval(0, 2))
    assert f9.intersect(Interval(1, 2)) == Intersection(f9, Interval(1, 2))
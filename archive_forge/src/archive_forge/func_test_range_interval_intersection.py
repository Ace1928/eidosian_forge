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
def test_range_interval_intersection():
    p = symbols('p', positive=True)
    assert isinstance(Range(3).intersect(Interval(p, p + 2)), Intersection)
    assert Range(4).intersect(Interval(0, 3)) == Range(4)
    assert Range(4).intersect(Interval(-oo, oo)) == Range(4)
    assert Range(4).intersect(Interval(1, oo)) == Range(1, 4)
    assert Range(4).intersect(Interval(1.1, oo)) == Range(2, 4)
    assert Range(4).intersect(Interval(0.1, 3)) == Range(1, 4)
    assert Range(4).intersect(Interval(0.1, 3.1)) == Range(1, 4)
    assert Range(4).intersect(Interval.open(0, 3)) == Range(1, 3)
    assert Range(4).intersect(Interval.open(0.1, 0.5)) is S.EmptySet
    assert Interval(-1, 5).intersect(S.Complexes) == Interval(-1, 5)
    assert Interval(-1, 5).intersect(S.Reals) == Interval(-1, 5)
    assert Interval(-1, 5).intersect(S.Integers) == Range(-1, 6)
    assert Interval(-1, 5).intersect(S.Naturals) == Range(1, 6)
    assert Interval(-1, 5).intersect(S.Naturals0) == Range(0, 6)
    assert Range(0).intersect(Interval(0.2, 0.8)) is S.EmptySet
    assert Range(0).intersect(Interval(-oo, oo)) is S.EmptySet
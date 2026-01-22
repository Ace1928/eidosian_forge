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
def test_ImageSet():
    raises(ValueError, lambda: ImageSet(x, S.Integers))
    assert ImageSet(Lambda(x, 1), S.Integers) == FiniteSet(1)
    assert ImageSet(Lambda(x, y), S.Integers) == {y}
    assert ImageSet(Lambda(x, 1), S.EmptySet) == S.EmptySet
    empty = Intersection(FiniteSet(log(2) / pi), S.Integers)
    assert unchanged(ImageSet, Lambda(x, 1), empty)
    squares = ImageSet(Lambda(x, x ** 2), S.Naturals)
    assert 4 in squares
    assert 5 not in squares
    assert FiniteSet(*range(10)).intersect(squares) == FiniteSet(1, 4, 9)
    assert 16 not in squares.intersect(Interval(0, 10))
    si = iter(squares)
    a, b, c, d = (next(si), next(si), next(si), next(si))
    assert (a, b, c, d) == (1, 4, 9, 16)
    harmonics = ImageSet(Lambda(x, 1 / x), S.Naturals)
    assert Rational(1, 5) in harmonics
    assert Rational(0.25) in harmonics
    assert 0.25 not in harmonics
    assert Rational(0.3) not in harmonics
    assert (1, 2) not in harmonics
    assert harmonics.is_iterable
    assert imageset(x, -x, Interval(0, 1)) == Interval(-1, 0)
    assert ImageSet(Lambda(x, x ** 2), Interval(0, 2)).doit() == Interval(0, 4)
    assert ImageSet(Lambda((x, y), 2 * x), {4}, {3}).doit() == FiniteSet(8)
    assert ImageSet(Lambda((x, y), x + y), {1, 2, 3}, {10, 20, 30}).doit() == FiniteSet(11, 12, 13, 21, 22, 23, 31, 32, 33)
    c = Interval(1, 3) * Interval(1, 3)
    assert Tuple(2, 6) in ImageSet(Lambda(((x, y),), (x, 2 * y)), c)
    assert Tuple(2, S.Half) in ImageSet(Lambda(((x, y),), (x, 1 / y)), c)
    assert Tuple(2, -2) not in ImageSet(Lambda(((x, y),), (x, y ** 2)), c)
    assert Tuple(2, -2) in ImageSet(Lambda(((x, y),), (x, -2)), c)
    c3 = ProductSet(Interval(3, 7), Interval(8, 11), Interval(5, 9))
    assert Tuple(8, 3, 9) in ImageSet(Lambda(((t, y, x),), (y, t, x)), c3)
    assert Tuple(Rational(1, 8), 3, 9) in ImageSet(Lambda(((t, y, x),), (1 / y, t, x)), c3)
    assert 2 / pi not in ImageSet(Lambda(((x, y),), 2 / x), c)
    assert 2 / S(100) not in ImageSet(Lambda(((x, y),), 2 / x), c)
    assert Rational(2, 3) in ImageSet(Lambda(((x, y),), 2 / x), c)
    S1 = imageset(lambda x, y: x + y, S.Integers, S.Naturals)
    assert S1.base_pset == ProductSet(S.Integers, S.Naturals)
    assert S1.base_sets == (S.Integers, S.Naturals)
    assert unchanged(ImageSet, Lambda(x, x ** 2), {1, 2, 3})
    S2 = ImageSet(Lambda(((x, y),), x + y), {(1, 2), (3, 4)})
    assert 3 in S2.doit()
    assert S2._contains(3) is None
    raises(TypeError, lambda: ImageSet(Lambda(x, x ** 2), 1))
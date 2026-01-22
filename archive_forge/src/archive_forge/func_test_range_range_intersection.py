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
def test_range_range_intersection():
    for a, b, r in [(Range(0), Range(1), S.EmptySet), (Range(3), Range(4, oo), S.EmptySet), (Range(3), Range(-3, -1), S.EmptySet), (Range(1, 3), Range(0, 3), Range(1, 3)), (Range(1, 3), Range(1, 4), Range(1, 3)), (Range(1, oo, 2), Range(2, oo, 2), S.EmptySet), (Range(0, oo, 2), Range(oo), Range(0, oo, 2)), (Range(0, oo, 2), Range(100), Range(0, 100, 2)), (Range(2, oo, 2), Range(oo), Range(2, oo, 2)), (Range(0, oo, 2), Range(5, 6), S.EmptySet), (Range(2, 80, 1), Range(55, 71, 4), Range(55, 71, 4)), (Range(0, 6, 3), Range(-oo, 5, 3), S.EmptySet), (Range(0, oo, 2), Range(5, oo, 3), Range(8, oo, 6)), (Range(4, 6, 2), Range(2, 16, 7), S.EmptySet)]:
        assert a.intersect(b) == r
        assert a.intersect(b.reversed) == r
        assert a.reversed.intersect(b) == r
        assert a.reversed.intersect(b.reversed) == r
        a, b = (b, a)
        assert a.intersect(b) == r
        assert a.intersect(b.reversed) == r
        assert a.reversed.intersect(b) == r
        assert a.reversed.intersect(b.reversed) == r
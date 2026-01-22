from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (false, true)
from sympy.matrices.common import MatrixKind
from sympy.matrices.dense import Matrix
from sympy.polys.rootoftools import rootof
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range)
from sympy.sets.sets import (Complement, DisjointUnion, FiniteSet, Intersection, Interval, ProductSet, Set, SymmetricDifference, Union, imageset, SetKind)
from mpmath import mpi
from sympy.core.expr import unchanged
from sympy.core.relational import Eq, Ne, Le, Lt, LessThan
from sympy.logic import And, Or, Xor
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.abc import x, y, z, m, n
def test_is_superset():
    assert Interval(0, 1).is_superset(Interval(0, 2)) == False
    assert Interval(0, 3).is_superset(Interval(0, 2))
    assert FiniteSet(1, 2).is_superset(FiniteSet(1, 2, 3, 4)) == False
    assert FiniteSet(4, 5).is_superset(FiniteSet(1, 2, 3, 4)) == False
    assert FiniteSet(1).is_superset(Interval(0, 2)) == False
    assert FiniteSet(1, 2).is_superset(Interval(0, 2, True, True)) == False
    assert (Interval(1, 2) + FiniteSet(3)).is_superset(Interval(0, 2, False, True) + FiniteSet(2, 3)) == False
    assert Interval(3, 4).is_superset(Union(Interval(0, 1), Interval(2, 5))) == False
    assert FiniteSet(1, 2, 3, 4).is_superset(Interval(0, 5)) == False
    assert S.EmptySet.is_superset(FiniteSet(1, 2, 3)) == False
    assert Interval(0, 1).is_superset(S.EmptySet) == True
    assert S.EmptySet.is_superset(S.EmptySet) == True
    raises(ValueError, lambda: S.EmptySet.is_superset(1))
    assert Interval(0, 1).issuperset(S.EmptySet) == True
    assert S.EmptySet.issuperset(S.EmptySet) == True
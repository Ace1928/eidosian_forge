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
def test_interval_arguments():
    assert Interval(0, oo) == Interval(0, oo, False, True)
    assert Interval(0, oo).right_open is true
    assert Interval(-oo, 0) == Interval(-oo, 0, True, False)
    assert Interval(-oo, 0).left_open is true
    assert Interval(oo, -oo) == S.EmptySet
    assert Interval(oo, oo) == S.EmptySet
    assert Interval(-oo, -oo) == S.EmptySet
    assert Interval(oo, x) == S.EmptySet
    assert Interval(oo, oo) == S.EmptySet
    assert Interval(x, -oo) == S.EmptySet
    assert Interval(x, x) == {x}
    assert isinstance(Interval(1, 1), FiniteSet)
    e = Sum(x, (x, 1, 3))
    assert isinstance(Interval(e, e), FiniteSet)
    assert Interval(1, 0) == S.EmptySet
    assert Interval(1, 1).measure == 0
    assert Interval(1, 1, False, True) == S.EmptySet
    assert Interval(1, 1, True, False) == S.EmptySet
    assert Interval(1, 1, True, True) == S.EmptySet
    assert isinstance(Interval(0, Symbol('a')), Interval)
    assert Interval(Symbol('a', positive=True), 0) == S.EmptySet
    raises(ValueError, lambda: Interval(0, S.ImaginaryUnit))
    raises(ValueError, lambda: Interval(0, Symbol('z', extended_real=False)))
    raises(ValueError, lambda: Interval(x, x + S.ImaginaryUnit))
    raises(NotImplementedError, lambda: Interval(0, 1, And(x, y)))
    raises(NotImplementedError, lambda: Interval(0, 1, False, And(x, y)))
    raises(NotImplementedError, lambda: Interval(0, 1, z, And(x, y)))
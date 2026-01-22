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
def test_supinf():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert (Interval(0, 1) + FiniteSet(2)).sup == 2
    assert (Interval(0, 1) + FiniteSet(2)).inf == 0
    assert (Interval(0, 1) + FiniteSet(x)).sup == Max(1, x)
    assert (Interval(0, 1) + FiniteSet(x)).inf == Min(0, x)
    assert FiniteSet(5, 1, x).sup == Max(5, x)
    assert FiniteSet(5, 1, x).inf == Min(1, x)
    assert FiniteSet(5, 1, x, y).sup == Max(5, x, y)
    assert FiniteSet(5, 1, x, y).inf == Min(1, x, y)
    assert FiniteSet(5, 1, x, y, S.Infinity, S.NegativeInfinity).sup == S.Infinity
    assert FiniteSet(5, 1, x, y, S.Infinity, S.NegativeInfinity).inf == S.NegativeInfinity
    assert FiniteSet('Ham', 'Eggs').sup == Max('Ham', 'Eggs')
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
def test_SetKind_evaluate_False():
    U = lambda *args: Union(*args, evaluate=False)
    assert U({1}, EmptySet).kind is SetKind(NumberKind)
    assert U(Interval(1, 2), EmptySet).kind is SetKind(NumberKind)
    assert U({1}, S.UniversalSet).kind is SetKind(UndefinedKind)
    assert U(Interval(1, 2), Interval(4, 5), FiniteSet(1)).kind is SetKind(NumberKind)
    I = lambda *args: Intersection(*args, evaluate=False)
    assert I({1}, S.UniversalSet).kind is SetKind(NumberKind)
    assert I({1}, EmptySet).kind is SetKind()
    C = lambda *args: Complement(*args, evaluate=False)
    assert C(S.UniversalSet, {1, 2, 4, 5}).kind is SetKind(UndefinedKind)
    assert C({1, 2, 3, 4, 5}, EmptySet).kind is SetKind(NumberKind)
    assert C(EmptySet, {1, 2, 3, 4, 5}).kind is SetKind()
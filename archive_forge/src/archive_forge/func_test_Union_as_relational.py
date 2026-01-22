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
def test_Union_as_relational():
    x = Symbol('x')
    assert (Interval(0, 1) + FiniteSet(2)).as_relational(x) == Or(And(Le(0, x), Le(x, 1)), Eq(x, 2))
    assert (Interval(0, 1, True, True) + FiniteSet(1)).as_relational(x) == And(Lt(0, x), Le(x, 1))
    assert Or(x < 0, x > 0).as_set().as_relational(x) == And(x > -oo, x < oo, Ne(x, 0))
    assert (Interval.Ropen(1, 3) + Interval.Lopen(3, 5)).as_relational(x) == And(Ne(x, 3), x >= 1, x <= 5)
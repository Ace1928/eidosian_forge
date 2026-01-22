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
def test_set_operations_nonsets():
    """Tests that e.g. FiniteSet(1) * 2 raises TypeError"""
    ops = [lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b, lambda a, b: a / b, lambda a, b: a // b, lambda a, b: a | b, lambda a, b: a & b, lambda a, b: a ^ b]
    Sx = FiniteSet(x)
    Sy = FiniteSet(y)
    sets = [{1}, FiniteSet(1), Interval(1, 2), Union(Sx, Interval(1, 2)), Intersection(Sx, Sy), Complement(Sx, Sy), ProductSet(Sx, Sy), S.EmptySet]
    nums = [0, 1, 2, S(0), S(1), S(2)]
    for si in sets:
        for ni in nums:
            for op in ops:
                raises(TypeError, lambda: op(si, ni))
                raises(TypeError, lambda: op(ni, si))
        raises(TypeError, lambda: si ** object())
        raises(TypeError, lambda: si ** {1})
from sympy.core import symbols, Symbol, Tuple, oo, Dummy
from sympy.tensor.indexed import IndexException
from sympy.testing.pytest import raises
from sympy.utilities.iterables import iterable
from sympy.concrete.summations import Sum
from sympy.core.function import Function, Subs, Derivative
from sympy.core.relational import (StrictLessThan, GreaterThan,
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.series.order import Order
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import IndexedBase, Idx, Indexed
def test_Subs_with_Indexed():
    A = IndexedBase('A')
    i, j, k = symbols('i,j,k')
    x, y, z = symbols('x,y,z')
    f = Function('f')
    assert Subs(A[i], A[i], A[j]).diff(A[j]) == 1
    assert Subs(A[i], A[i], x).diff(A[i]) == 0
    assert Subs(A[i], A[i], x).diff(A[j]) == 0
    assert Subs(A[i], A[i], x).diff(x) == 1
    assert Subs(A[i], A[i], x).diff(y) == 0
    assert Subs(A[i], A[i], A[j]).diff(A[k]) == KroneckerDelta(j, k)
    assert Subs(x, x, A[i]).diff(A[j]) == KroneckerDelta(i, j)
    assert Subs(f(A[i]), A[i], x).diff(A[j]) == 0
    assert Subs(f(A[i]), A[i], A[k]).diff(A[j]) == Derivative(f(A[k]), A[k]) * KroneckerDelta(j, k)
    assert Subs(x, x, A[i] ** 2).diff(A[j]) == 2 * KroneckerDelta(i, j) * A[i]
    assert Subs(A[i], A[i], A[j] ** 2).diff(A[k]) == 2 * KroneckerDelta(j, k) * A[j]
    assert Subs(A[i] * x, x, A[i]).diff(A[i]) == 2 * A[i]
    assert Subs(A[i] * x, x, A[i]).diff(A[j]) == 2 * A[i] * KroneckerDelta(i, j)
    assert Subs(A[i] * x, x, A[j]).diff(A[i]) == A[j] + A[i] * KroneckerDelta(i, j)
    assert Subs(A[i] * x, x, A[j]).diff(A[j]) == A[i] + A[j] * KroneckerDelta(i, j)
    assert Subs(A[i] * x, x, A[i]).diff(A[k]) == 2 * A[i] * KroneckerDelta(i, k)
    assert Subs(A[i] * x, x, A[j]).diff(A[k]) == KroneckerDelta(i, k) * A[j] + KroneckerDelta(j, k) * A[i]
    assert Subs(A[i] * x, A[i], x).diff(A[i]) == 0
    assert Subs(A[i] * x, A[i], x).diff(A[j]) == 0
    assert Subs(A[i] * x, A[j], x).diff(A[i]) == x
    assert Subs(A[i] * x, A[j], x).diff(A[j]) == x * KroneckerDelta(i, j)
    assert Subs(A[i] * x, A[i], x).diff(A[k]) == 0
    assert Subs(A[i] * x, A[j], x).diff(A[k]) == x * KroneckerDelta(i, k)
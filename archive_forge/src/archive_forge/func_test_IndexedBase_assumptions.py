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
def test_IndexedBase_assumptions():
    i = Symbol('i', integer=True)
    a = Symbol('a')
    A = IndexedBase(a, positive=True)
    for c in (A, A[i]):
        assert c.is_real
        assert c.is_complex
        assert not c.is_imaginary
        assert c.is_nonnegative
        assert c.is_nonzero
        assert c.is_commutative
        assert log(exp(c)) == c
    assert A != IndexedBase(a)
    assert A == IndexedBase(a, positive=True, real=True)
    assert A[i] != Indexed(a, i)
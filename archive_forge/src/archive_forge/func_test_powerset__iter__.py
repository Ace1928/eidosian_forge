from sympy.core.expr import unchanged
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Interval
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import FiniteSet
from sympy.testing.pytest import raises, XFAIL
def test_powerset__iter__():
    a = PowerSet(FiniteSet(1, 2)).__iter__()
    assert next(a) == S.EmptySet
    assert next(a) == FiniteSet(1)
    assert next(a) == FiniteSet(2)
    assert next(a) == FiniteSet(1, 2)
    a = PowerSet(S.Naturals).__iter__()
    assert next(a) == S.EmptySet
    assert next(a) == FiniteSet(1)
    assert next(a) == FiniteSet(2)
    assert next(a) == FiniteSet(1, 2)
    assert next(a) == FiniteSet(3)
    assert next(a) == FiniteSet(1, 3)
    assert next(a) == FiniteSet(2, 3)
    assert next(a) == FiniteSet(1, 2, 3)
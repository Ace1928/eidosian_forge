from sympy.core.expr import unchanged
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Interval
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import FiniteSet
from sympy.testing.pytest import raises, XFAIL
def test_powerset__len__():
    A = PowerSet(S.EmptySet, evaluate=False)
    assert len(A) == 1
    A = PowerSet(A, evaluate=False)
    assert len(A) == 2
    A = PowerSet(A, evaluate=False)
    assert len(A) == 4
    A = PowerSet(A, evaluate=False)
    assert len(A) == 16
from sympy.core.expr import unchanged
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Interval
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import FiniteSet
from sympy.testing.pytest import raises, XFAIL
def test_powerset_method():
    A = FiniteSet()
    pset = A.powerset()
    assert len(pset) == 1
    assert pset == FiniteSet(S.EmptySet)
    A = FiniteSet(1, 2)
    pset = A.powerset()
    assert len(pset) == 2 ** len(A)
    assert pset == FiniteSet(FiniteSet(), FiniteSet(1), FiniteSet(2), A)
    A = Interval(0, 1)
    assert A.powerset() == PowerSet(A)
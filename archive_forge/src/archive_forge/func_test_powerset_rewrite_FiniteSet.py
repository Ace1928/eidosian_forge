from sympy.core.expr import unchanged
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Interval
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import FiniteSet
from sympy.testing.pytest import raises, XFAIL
def test_powerset_rewrite_FiniteSet():
    assert PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet) == FiniteSet(S.EmptySet, FiniteSet(1), FiniteSet(2), FiniteSet(1, 2))
    assert PowerSet(S.EmptySet).rewrite(FiniteSet) == FiniteSet(S.EmptySet)
    assert PowerSet(S.Naturals).rewrite(FiniteSet) == PowerSet(S.Naturals)
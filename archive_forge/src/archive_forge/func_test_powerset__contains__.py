from sympy.core.expr import unchanged
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Interval
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import FiniteSet
from sympy.testing.pytest import raises, XFAIL
def test_powerset__contains__():
    subset_series = [S.EmptySet, FiniteSet(1, 2), S.Naturals, S.Naturals0, S.Integers, S.Rationals, S.Reals, S.Complexes]
    l = len(subset_series)
    for i in range(l):
        for j in range(l):
            if i <= j:
                assert subset_series[i] in PowerSet(subset_series[j], evaluate=False)
            else:
                assert subset_series[i] not in PowerSet(subset_series[j], evaluate=False)
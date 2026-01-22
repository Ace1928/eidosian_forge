from sympy.core.expr import unchanged
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Interval
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import FiniteSet
from sympy.testing.pytest import raises, XFAIL
def test_powerset_contains():
    A = PowerSet(FiniteSet(1), evaluate=False)
    assert A.contains(2) == Contains(2, A)
    x = Symbol('x')
    A = PowerSet(FiniteSet(x), evaluate=False)
    assert A.contains(FiniteSet(1)) == Contains(FiniteSet(1), A)
from sympy.core.expr import unchanged
from sympy.core.numbers import oo
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.testing.pytest import raises
def test_as_set():
    x = Symbol('x')
    y = Symbol('y')
    assert Contains(x, FiniteSet(y)).as_set() == FiniteSet(y)
    assert Contains(x, S.Integers).as_set() == S.Integers
    assert Contains(x, S.Reals).as_set() == S.Reals
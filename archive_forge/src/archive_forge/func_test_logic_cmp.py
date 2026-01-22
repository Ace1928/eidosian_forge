from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
from sympy.testing.pytest import raises
from itertools import product
def test_logic_cmp():
    l1 = And('a', Not('b'))
    l2 = And('a', Not('b'))
    assert hash(l1) == hash(l2)
    assert (l1 == l2) == T
    assert (l1 != l2) == F
    assert And('a', 'b', 'c') == And('b', 'a', 'c')
    assert And('a', 'b', 'c') == And('c', 'b', 'a')
    assert And('a', 'b', 'c') == And('c', 'a', 'b')
    assert Not('a') < Not('b')
    assert (Not('b') < Not('a')) is False
    assert (Not('a') < 2) is False
from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
from sympy.testing.pytest import raises
from itertools import product
def test_logic_combine_args():
    assert And('a', 'b', 'a') == And('a', 'b')
    assert Or('a', 'b', 'a') == Or('a', 'b')
    assert And(And('a', 'b'), And('c', 'd')) == And('a', 'b', 'c', 'd')
    assert Or(Or('a', 'b'), Or('c', 'd')) == Or('a', 'b', 'c', 'd')
    assert Or('t', And('n', 'p', 'r'), And('n', 'r'), And('n', 'p', 'r'), 't', And('n', 'r')) == Or('t', And('n', 'p', 'r'), And('n', 'r'))
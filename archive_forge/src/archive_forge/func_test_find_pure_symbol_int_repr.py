from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_find_pure_symbol_int_repr():
    assert find_pure_symbol_int_repr([1], [{1}]) == (1, True)
    assert find_pure_symbol_int_repr([1, 2], [{-1, 2}, {-2, 1}]) == (None, None)
    assert find_pure_symbol_int_repr([1, 2, 3], [{1, -2}, {-2, -3}, {3, 1}]) == (1, True)
    assert find_pure_symbol_int_repr([1, 2, 3], [{-1, 2}, {2, -3}, {3, 1}]) == (2, True)
    assert find_pure_symbol_int_repr([1, 2, 3], [{-1, -2}, {-2, -3}, {3, 1}]) == (2, False)
    assert find_pure_symbol_int_repr([1, 2, 3], [{-1, 2}, {-2, -3}, {3, 1}]) == (None, None)
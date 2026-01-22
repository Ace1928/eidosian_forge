from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_unit_propagate_int_repr():
    assert unit_propagate_int_repr([{1, 2}], 1) == []
    assert unit_propagate_int_repr(map(set, [[1, 2], [-1, 3], [-3, 2], [1]]), 1) == [{3}, {-3, 2}]
from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_pl_true_wrong_input():
    from sympy.core.numbers import pi
    raises(ValueError, lambda: pl_true('John Cleese'))
    raises(ValueError, lambda: pl_true(42 + pi + pi ** 2))
    raises(ValueError, lambda: pl_true(42))
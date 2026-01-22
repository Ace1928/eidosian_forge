from sympy.logic.utilities.dimacs import load
from sympy.logic.algorithms.dpll import dpll_satisfiable
def test_f4():
    assert not bool(dpll_satisfiable(load(f4)))
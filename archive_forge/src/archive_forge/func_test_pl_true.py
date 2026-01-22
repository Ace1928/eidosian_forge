from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_pl_true():
    A, B, C = symbols('A,B,C')
    assert pl_true(True) is True
    assert pl_true(A & B, {A: True, B: True}) is True
    assert pl_true(A | B, {A: True}) is True
    assert pl_true(A | B, {B: True}) is True
    assert pl_true(A | B, {A: None, B: True}) is True
    assert pl_true(A >> B, {A: False}) is True
    assert pl_true(A | B | ~C, {A: False, B: True, C: True}) is True
    assert pl_true(Equivalent(A, B), {A: False, B: False}) is True
    assert pl_true(False) is False
    assert pl_true(A & B, {A: False, B: False}) is False
    assert pl_true(A & B, {A: False}) is False
    assert pl_true(A & B, {B: False}) is False
    assert pl_true(A | B, {A: False, B: False}) is False
    assert pl_true(B, {B: None}) is None
    assert pl_true(A & B, {A: True, B: None}) is None
    assert pl_true(A >> B, {A: True, B: None}) is None
    assert pl_true(Equivalent(A, B), {A: None}) is None
    assert pl_true(Equivalent(A, B), {A: True, B: None}) is None
    assert pl_true(A | B, {A: False}, deep=True) is None
    assert pl_true(~A & ~B, {A: False}, deep=True) is None
    assert pl_true(A | B, {A: False, B: False}, deep=True) is False
    assert pl_true(A & B & (~A | ~B), {A: True}, deep=True) is False
    assert pl_true(C >> A >> (B >> A), {C: True}, deep=True) is True
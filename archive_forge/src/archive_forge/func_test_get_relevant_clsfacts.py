from sympy.assumptions.ask import Q
from sympy.assumptions.assume import assuming
from sympy.core.numbers import (I, pi)
from sympy.core.relational import (Eq, Gt)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import Abs
from sympy.logic.boolalg import Implies
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.assumptions.cnf import CNF, Literal
from sympy.assumptions.satask import (satask, extract_predargs,
from sympy.testing.pytest import raises, XFAIL
def test_get_relevant_clsfacts():
    exprs = {Abs(x * y)}
    exprs, facts = get_relevant_clsfacts(exprs)
    assert exprs == {x * y}
    assert facts.clauses == {frozenset({Literal(Q.odd(Abs(x * y)), False), Literal(Q.odd(x * y), True)}), frozenset({Literal(Q.zero(Abs(x * y)), False), Literal(Q.zero(x * y), True)}), frozenset({Literal(Q.even(Abs(x * y)), False), Literal(Q.even(x * y), True)}), frozenset({Literal(Q.zero(Abs(x * y)), True), Literal(Q.zero(x * y), False)}), frozenset({Literal(Q.even(Abs(x * y)), False), Literal(Q.odd(Abs(x * y)), False), Literal(Q.odd(x * y), True)}), frozenset({Literal(Q.even(Abs(x * y)), False), Literal(Q.even(x * y), True), Literal(Q.odd(Abs(x * y)), False)}), frozenset({Literal(Q.positive(Abs(x * y)), False), Literal(Q.zero(Abs(x * y)), False)})}
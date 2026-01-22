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
def test_pos_neg():
    assert satask(~Q.positive(x), Q.negative(x)) is True
    assert satask(~Q.negative(x), Q.positive(x)) is True
    assert satask(Q.positive(x + y), Q.positive(x) & Q.positive(y)) is True
    assert satask(Q.negative(x + y), Q.negative(x) & Q.negative(y)) is True
    assert satask(Q.positive(x + y), Q.negative(x) & Q.negative(y)) is False
    assert satask(Q.negative(x + y), Q.positive(x) & Q.positive(y)) is False
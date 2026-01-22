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
def test_even_satask():
    assert satask(Q.even(2)) is True
    assert satask(Q.even(3)) is False
    assert satask(Q.even(x * y), Q.even(x) & Q.odd(y)) is True
    assert satask(Q.even(x * y), Q.even(x) & Q.integer(y)) is True
    assert satask(Q.even(x * y), Q.even(x) & Q.even(y)) is True
    assert satask(Q.even(x * y), Q.odd(x) & Q.odd(y)) is False
    assert satask(Q.even(x * y), Q.even(x)) is None
    assert satask(Q.even(x * y), Q.odd(x) & Q.integer(y)) is None
    assert satask(Q.even(x * y), Q.odd(x) & Q.odd(y)) is False
    assert satask(Q.even(abs(x)), Q.even(x)) is True
    assert satask(Q.even(abs(x)), Q.odd(x)) is False
    assert satask(Q.even(x), Q.even(abs(x))) is None
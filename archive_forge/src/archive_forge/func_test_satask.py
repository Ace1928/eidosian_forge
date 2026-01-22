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
def test_satask():
    assert satask(Q.real(x), Q.real(x)) is True
    assert satask(Q.real(x), ~Q.real(x)) is False
    assert satask(Q.real(x)) is None
    assert satask(Q.real(x), Q.positive(x)) is True
    assert satask(Q.positive(x), Q.real(x)) is None
    assert satask(Q.real(x), ~Q.positive(x)) is None
    assert satask(Q.positive(x), ~Q.real(x)) is False
    raises(ValueError, lambda: satask(Q.real(x), Q.real(x) & ~Q.real(x)))
    with assuming(Q.positive(x)):
        assert satask(Q.real(x)) is True
        assert satask(~Q.positive(x)) is False
        raises(ValueError, lambda: satask(Q.real(x), ~Q.positive(x)))
    assert satask(Q.zero(x), Q.nonzero(x)) is False
    assert satask(Q.positive(x), Q.zero(x)) is False
    assert satask(Q.real(x), Q.zero(x)) is True
    assert satask(Q.zero(x), Q.zero(x * y)) is None
    assert satask(Q.zero(x * y), Q.zero(x))
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
def test_pow_pos_neg():
    assert satask(Q.nonnegative(x ** 2), Q.positive(x)) is True
    assert satask(Q.nonpositive(x ** 2), Q.positive(x)) is False
    assert satask(Q.positive(x ** 2), Q.positive(x)) is True
    assert satask(Q.negative(x ** 2), Q.positive(x)) is False
    assert satask(Q.real(x ** 2), Q.positive(x)) is True
    assert satask(Q.nonnegative(x ** 2), Q.negative(x)) is True
    assert satask(Q.nonpositive(x ** 2), Q.negative(x)) is False
    assert satask(Q.positive(x ** 2), Q.negative(x)) is True
    assert satask(Q.negative(x ** 2), Q.negative(x)) is False
    assert satask(Q.real(x ** 2), Q.negative(x)) is True
    assert satask(Q.nonnegative(x ** 2), Q.nonnegative(x)) is True
    assert satask(Q.nonpositive(x ** 2), Q.nonnegative(x)) is None
    assert satask(Q.positive(x ** 2), Q.nonnegative(x)) is None
    assert satask(Q.negative(x ** 2), Q.nonnegative(x)) is False
    assert satask(Q.real(x ** 2), Q.nonnegative(x)) is True
    assert satask(Q.nonnegative(x ** 2), Q.nonpositive(x)) is True
    assert satask(Q.nonpositive(x ** 2), Q.nonpositive(x)) is None
    assert satask(Q.positive(x ** 2), Q.nonpositive(x)) is None
    assert satask(Q.negative(x ** 2), Q.nonpositive(x)) is False
    assert satask(Q.real(x ** 2), Q.nonpositive(x)) is True
    assert satask(Q.nonnegative(x ** 3), Q.positive(x)) is True
    assert satask(Q.nonpositive(x ** 3), Q.positive(x)) is False
    assert satask(Q.positive(x ** 3), Q.positive(x)) is True
    assert satask(Q.negative(x ** 3), Q.positive(x)) is False
    assert satask(Q.real(x ** 3), Q.positive(x)) is True
    assert satask(Q.nonnegative(x ** 3), Q.negative(x)) is False
    assert satask(Q.nonpositive(x ** 3), Q.negative(x)) is True
    assert satask(Q.positive(x ** 3), Q.negative(x)) is False
    assert satask(Q.negative(x ** 3), Q.negative(x)) is True
    assert satask(Q.real(x ** 3), Q.negative(x)) is True
    assert satask(Q.nonnegative(x ** 3), Q.nonnegative(x)) is True
    assert satask(Q.nonpositive(x ** 3), Q.nonnegative(x)) is None
    assert satask(Q.positive(x ** 3), Q.nonnegative(x)) is None
    assert satask(Q.negative(x ** 3), Q.nonnegative(x)) is False
    assert satask(Q.real(x ** 3), Q.nonnegative(x)) is True
    assert satask(Q.nonnegative(x ** 3), Q.nonpositive(x)) is None
    assert satask(Q.nonpositive(x ** 3), Q.nonpositive(x)) is True
    assert satask(Q.positive(x ** 3), Q.nonpositive(x)) is False
    assert satask(Q.negative(x ** 3), Q.nonpositive(x)) is None
    assert satask(Q.real(x ** 3), Q.nonpositive(x)) is True
    assert satask(Q.nonnegative(x ** (-2)), Q.nonpositive(x)) is None
    assert satask(Q.nonpositive(x ** (-2)), Q.nonpositive(x)) is None
    assert satask(Q.positive(x ** (-2)), Q.nonpositive(x)) is None
    assert satask(Q.negative(x ** (-2)), Q.nonpositive(x)) is None
    assert satask(Q.real(x ** (-2)), Q.nonpositive(x)) is None
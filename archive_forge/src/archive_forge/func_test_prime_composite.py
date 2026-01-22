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
def test_prime_composite():
    assert satask(Q.prime(x), Q.composite(x)) is False
    assert satask(Q.composite(x), Q.prime(x)) is False
    assert satask(Q.composite(x), ~Q.prime(x)) is None
    assert satask(Q.prime(x), ~Q.composite(x)) is None
    assert satask(Q.prime(x), Q.integer(x) & Q.positive(x) & ~Q.composite(x)) is None
    assert satask(Q.prime(2)) is True
    assert satask(Q.prime(4)) is False
    assert satask(Q.prime(1)) is False
    assert satask(Q.composite(1)) is False
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.polys.polytools import PurePoly
from sympy.matrices import \
from sympy.testing.pytest import raises
def test_CL_RL():
    assert SparseMatrix(((1, 2), (3, 4))).row_list() == [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 4)]
    assert SparseMatrix(((1, 2), (3, 4))).col_list() == [(0, 0, 1), (1, 0, 3), (0, 1, 2), (1, 1, 4)]
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.polys.polytools import PurePoly
from sympy.matrices import \
from sympy.testing.pytest import raises
def test_diagonal_solve():
    a, d = symbols('a d')
    u, v, w, x = symbols('u:x')
    A = SparseMatrix([[a, 0], [0, d]])
    B = MutableSparseMatrix([[u, v], [w, x]])
    C = ImmutableSparseMatrix([[u, v], [w, x]])
    sol = Matrix([[u / a, v / a], [w / d, x / d]])
    assert A.diagonal_solve(B) == sol
    assert A.diagonal_solve(C) == sol
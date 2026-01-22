from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_bifid5_square():
    A = bifid5
    f = lambda i, j: symbols(A[5 * i + j])
    M = Matrix(5, 5, f)
    assert bifid5_square('') == M
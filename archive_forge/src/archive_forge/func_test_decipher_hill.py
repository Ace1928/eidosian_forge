from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_decipher_hill():
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert decipher_hill('CFIV', A) == 'ABCD'
    A = Matrix(2, 2, [1, 0, 0, 1])
    assert decipher_hill('ABCD', A) == 'ABCD'
    assert decipher_hill('ABCD', A, symbols='ABCD') == 'ABCD'
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert decipher_hill('CBAB', A, symbols='ABCD') == 'ABCD'
    assert decipher_hill('CB', A, symbols='ABCD') == 'AB'
    assert decipher_hill('CFA', A) == 'ABAA'
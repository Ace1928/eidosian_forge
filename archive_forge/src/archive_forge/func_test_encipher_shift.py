from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_encipher_shift():
    assert encipher_shift('ABC', 0) == 'ABC'
    assert encipher_shift('ABC', 1) == 'BCD'
    assert encipher_shift('ABC', -1) == 'ZAB'
    assert decipher_shift('ZAB', -1) == 'ABC'
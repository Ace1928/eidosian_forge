from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_encipher_atbash():
    assert encipher_atbash('ABC') == 'ZYX'
    assert encipher_atbash('ZYX') == 'ABC'
    assert decipher_atbash('ABC') == 'ZYX'
    assert decipher_atbash('ZYX') == 'ABC'
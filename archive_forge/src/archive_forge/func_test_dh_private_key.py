from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_dh_private_key():
    p, g, _ = dh_private_key(digit=100)
    assert isprime(p)
    assert is_primitive_root(g, p)
    assert len(bin(p)) >= 102
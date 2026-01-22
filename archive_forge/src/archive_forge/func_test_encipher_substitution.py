from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_encipher_substitution():
    assert encipher_substitution('ABC', 'BAC', 'ABC') == 'BAC'
    assert encipher_substitution('123', '1243', '1234') == '124'
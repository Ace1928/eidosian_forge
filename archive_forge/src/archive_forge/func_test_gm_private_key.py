from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_gm_private_key():
    raises(ValueError, lambda: gm_public_key(13, 15))
    raises(ValueError, lambda: gm_public_key(0, 0))
    raises(ValueError, lambda: gm_public_key(0, 5))
    assert 17, 19 == gm_public_key(17, 19)
from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange
def test_check_and_join():
    assert check_and_join('abc') == 'abc'
    assert check_and_join(uniq('aaabc')) == 'abc'
    assert check_and_join('ab c'.split()) == 'abc'
    assert check_and_join('abc', 'a', filter=True) == 'a'
    raises(ValueError, lambda: check_and_join('ab', 'a'))
from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_primitive_root_list():
    assert primitive_root_list(54) == [5, 11, 23, 29, 41, 47]
    assert primitive_root_list(12) == []
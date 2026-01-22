from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_kronecker():
    assert kronecker(9, 2) == 1
    assert kronecker(-5, -1) == -1
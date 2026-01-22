from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_quotient_mod():
    assert quotient_mod(13, 5) == (2, 3)
    assert quotient_mod(-4, 7) == (-1, 3)
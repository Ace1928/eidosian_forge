from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_sieve_iterator():
    it = Sieve_iterator(101)
    assert len([i for i in it]) == 26
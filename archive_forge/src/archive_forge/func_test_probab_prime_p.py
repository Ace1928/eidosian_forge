from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_probab_prime_p():
    s = set(Sieve.generate_primes(1000))
    for n in range(1001):
        assert (n in s) == isprime(n)
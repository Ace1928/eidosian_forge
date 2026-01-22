from Cryptodome import Random
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.py3compat import iter_range
from Cryptodome.Util.number import sieve_base as _sieve_base_large
def generate_probable_prime(**kwargs):
    """Generate a random probable prime.

    The prime will not have any specific properties
    (e.g. it will not be a *strong* prime).

    Random numbers are evaluated for primality until one
    passes all tests, consisting of a certain number of
    Miller-Rabin tests with random bases followed by
    a single Lucas test.

    The number of Miller-Rabin iterations is chosen such that
    the probability that the output number is a non-prime is
    less than 1E-30 (roughly 2^{-100}).

    This approach is compliant to `FIPS PUB 186-4`__.

    :Keywords:
      exact_bits : integer
        The desired size in bits of the probable prime.
        It must be at least 160.
      randfunc : callable
        An RNG function where candidate primes are taken from.
      prime_filter : callable
        A function that takes an Integer as parameter and returns
        True if the number can be passed to further primality tests,
        False if it should be immediately discarded.

    :Return:
        A probable prime in the range 2^exact_bits > p > 2^(exact_bits-1).

    .. __: http://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf
    """
    exact_bits = kwargs.pop('exact_bits', None)
    randfunc = kwargs.pop('randfunc', None)
    prime_filter = kwargs.pop('prime_filter', lambda x: True)
    if kwargs:
        raise ValueError('Unknown parameters: ' + kwargs.keys())
    if exact_bits is None:
        raise ValueError('Missing exact_bits parameter')
    if exact_bits < 160:
        raise ValueError('Prime number is not big enough.')
    if randfunc is None:
        randfunc = Random.new().read
    result = COMPOSITE
    while result == COMPOSITE:
        candidate = Integer.random(exact_bits=exact_bits, randfunc=randfunc) | 1
        if not prime_filter(candidate):
            continue
        result = test_probable_prime(candidate, randfunc)
    return candidate
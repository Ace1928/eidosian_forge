from Cryptodome import Random
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.py3compat import iter_range
from Cryptodome.Util.number import sieve_base as _sieve_base_large
def generate_probable_safe_prime(**kwargs):
    """Generate a random, probable safe prime.

    Note this operation is much slower than generating a simple prime.

    :Keywords:
      exact_bits : integer
        The desired size in bits of the probable safe prime.
      randfunc : callable
        An RNG function where candidate primes are taken from.

    :Return:
        A probable safe prime in the range
        2^exact_bits > p > 2^(exact_bits-1).
    """
    exact_bits = kwargs.pop('exact_bits', None)
    randfunc = kwargs.pop('randfunc', None)
    if kwargs:
        raise ValueError('Unknown parameters: ' + kwargs.keys())
    if randfunc is None:
        randfunc = Random.new().read
    result = COMPOSITE
    while result == COMPOSITE:
        q = generate_probable_prime(exact_bits=exact_bits - 1, randfunc=randfunc)
        candidate = q * 2 + 1
        if candidate.size_in_bits() != exact_bits:
            continue
        result = test_probable_prime(candidate, randfunc=randfunc)
    return candidate
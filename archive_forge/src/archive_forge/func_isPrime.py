import math
import sys
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import iter_range
import struct
import warnings
def isPrime(N, false_positive_prob=1e-06, randfunc=None):
    """Test if a number *N* is a prime.

    Args:
        false_positive_prob (float):
          The statistical probability for the result not to be actually a
          prime. It defaults to 10\\ :sup:`-6`.
          Note that the real probability of a false-positive is far less.
          This is just the mathematically provable limit.
        randfunc (callable):
          A function that takes a parameter *N* and that returns
          a random byte string of such length.
          If omitted, :func:`Cryptodome.Random.get_random_bytes` is used.

    Return:
        `True` is the input is indeed prime.
    """
    if randfunc is None:
        randfunc = Random.get_random_bytes
    if _fastmath is not None:
        return _fastmath.isPrime(long(N), false_positive_prob, randfunc)
    if N < 3 or N & 1 == 0:
        return N == 2
    for p in sieve_base:
        if N == p:
            return True
        if N % p == 0:
            return False
    rounds = int(math.ceil(-math.log(false_positive_prob) / math.log(4)))
    return bool(_rabinMillerTest(N, rounds, randfunc))
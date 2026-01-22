import math
import sys
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import iter_range
import struct
import warnings
def getPrime(N, randfunc=None):
    """Return a random N-bit prime number.

    N must be an integer larger than 1.
    If randfunc is omitted, then :meth:`Random.get_random_bytes` is used.
    """
    if randfunc is None:
        randfunc = Random.get_random_bytes
    if N < 2:
        raise ValueError('N must be larger than 1')
    while True:
        number = getRandomNBitInteger(N, randfunc) | 1
        if isPrime(number, randfunc=randfunc):
            break
    return number
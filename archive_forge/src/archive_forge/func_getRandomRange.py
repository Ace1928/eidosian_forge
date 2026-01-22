import math
import sys
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import iter_range
import struct
import warnings
def getRandomRange(a, b, randfunc=None):
    """Return a random number *n* so that *a <= n < b*.

    If :data:`randfunc` is omitted, then :meth:`Random.get_random_bytes` is used.

    .. deprecated:: 3.0
        This function is for internal use only and may be renamed or removed in
        the future. Use :func:`Cryptodome.Random.random.randrange` instead.
    """
    range_ = b - a - 1
    bits = size(range_)
    value = getRandomInteger(bits, randfunc)
    while value > range_:
        value = getRandomInteger(bits, randfunc)
    return a + value
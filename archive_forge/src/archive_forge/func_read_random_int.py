import os
from rsa import common, transform
from rsa._compat import byte
def read_random_int(nbits):
    """Reads a random integer of approximately nbits bits.
    """
    randomdata = read_random_bits(nbits)
    value = transform.bytes2int(randomdata)
    value |= 1 << nbits - 1
    return value
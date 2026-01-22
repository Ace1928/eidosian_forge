from __future__ import absolute_import
import itertools
import sys
from struct import pack
def xor_bytes(b1, b2):
    """
    Returns the bitwise XOR result between two bytes objects, b1 ^ b2.

    Bitwise XOR operation is commutative, so order of parameters doesn't
    generate different results. If parameters have different length, extra
    length of the largest one is ignored.

    :param b1:
        First bytes object.
    :param b2:
        Second bytes object.
    :returns:
        Bytes object, result of XOR operation.
    """
    if PY2:
        return ''.join((byte(ord(x) ^ ord(y)) for x, y in zip(b1, b2)))
    return bytes((x ^ y for x, y in zip(b1, b2)))
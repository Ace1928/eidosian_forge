import os
import zlib
import time  # noqa
import logging
import numpy as np
def signedint2bits(i, n=None):
    """convert signed int to a string of bits (0's and 1's in a string),
    pad to n elements. Negative numbers are stored in 2's complement bit
    patterns, thus positive numbers always start with a 0.
    """
    ii = i
    if i < 0:
        ii = abs(ii) - 1
    bb = BitArray()
    while ii > 0:
        bb += str(ii % 2)
        ii = ii >> 1
    bb.reverse()
    bb = '0' + str(bb)
    if n is not None:
        if len(bb) > n:
            raise ValueError('signedint2bits fail: len larger than padlength.')
        bb = bb.rjust(n, '0')
    if i < 0:
        bb = bb.replace('0', 'x').replace('1', '0').replace('x', '1')
    return BitArray(bb)
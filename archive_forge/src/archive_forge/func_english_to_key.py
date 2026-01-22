from __future__ import print_function
import binascii
from Cryptodome.Util.py3compat import bord, bchr
def english_to_key(s):
    """Transform a string into a corresponding key.

    Example::

        >>> from Cryptodome.Util.RFC1751 import english_to_key
        >>> english_to_key('RAM LOIS GOAD CREW CARE HIT')
        b'66666666'

    Args:
      s (string): the string with the words separated by whitespace;
                  the number of words must be a multiple of 6.
    Return:
      A byte string.
    """
    L = s.upper().split()
    key = b''
    for index in range(0, len(L), 6):
        sublist = L[index:index + 6]
        char = 9 * [0]
        bits = 0
        for i in sublist:
            index = wordlist.index(i)
            shift = (8 - (bits + 11) % 8) % 8
            y = index << shift
            cl, cc, cr = (y >> 16, y >> 8 & 255, y & 255)
            if shift > 5:
                char[bits >> 3] = char[bits >> 3] | cl
                char[(bits >> 3) + 1] = char[(bits >> 3) + 1] | cc
                char[(bits >> 3) + 2] = char[(bits >> 3) + 2] | cr
            elif shift > -3:
                char[bits >> 3] = char[bits >> 3] | cc
                char[(bits >> 3) + 1] = char[(bits >> 3) + 1] | cr
            else:
                char[bits >> 3] = char[bits >> 3] | cr
            bits = bits + 11
        subkey = b''
        for y in char:
            subkey = subkey + bchr(y)
        skbin = _key2bin(subkey)
        p = 0
        for i in range(0, 64, 2):
            p = p + _extract(skbin, i, 2)
        if p & 3 != _extract(skbin, 64, 2):
            raise ValueError('Parity error in resulting key')
        key = key + subkey[0:8]
    return key
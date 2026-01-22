import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
def packed_to_int(packed_int):
    """
    :param packed_int: a packed string containing an unsigned integer.
        It is assumed that string is packed in network byte order.

    :return: An unsigned integer equivalent to value of network address
        represented by packed binary string.
    """
    words = list(_struct.unpack('>8B', packed_int))
    int_val = 0
    for i, num in enumerate(reversed(words)):
        word = num
        word = word << 8 * i
        int_val = int_val | word
    return int_val
from __future__ import annotations
from struct import pack, unpack_from
def si16le(c: bytes, o: int=0) -> int:
    """
    Converts a 2-bytes (16 bits) string to a signed integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from('<h', c, o)[0]
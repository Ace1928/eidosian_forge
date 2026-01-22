from __future__ import annotations
from struct import pack, unpack_from
def si32le(c: bytes, o: int=0) -> int:
    """
    Converts a 4-bytes (32 bits) string to a signed integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from('<i', c, o)[0]
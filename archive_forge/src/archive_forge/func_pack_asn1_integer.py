import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def pack_asn1_integer(value: int, tag: bool=True) -> bytes:
    """Packs an int value into an ASN.1 INTEGER byte value with optional universal tagging."""
    is_negative = False
    limit = 127
    if value < 0:
        value = -value
        is_negative = True
        limit = 128
    b_int = bytearray()
    while value > limit:
        val = value & 255
        if is_negative:
            val = 255 - val
        b_int.append(val)
        value >>= 8
    b_int.append((255 - value if is_negative else value) & 255)
    if is_negative:
        for idx, val in enumerate(b_int):
            if val < 255:
                b_int[idx] += 1
                break
            b_int[idx] = 0
    if is_negative and b_int[-1] == 127:
        b_int.append(255)
    b_int.reverse()
    b_value = bytes(b_int)
    if tag:
        b_value = pack_asn1(TagClass.universal, False, TypeTagNumber.integer, b_value)
    return b_value
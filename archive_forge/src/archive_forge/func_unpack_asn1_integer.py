import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def unpack_asn1_integer(value: typing.Union[ASN1Value, bytes]) -> int:
    """Unpacks an ASN.1 INTEGER value."""
    b_int = bytearray(extract_asn1_tlv(value, TagClass.universal, TypeTagNumber.integer))
    is_negative = b_int[0] & 128
    if is_negative:
        for i in range(len(b_int)):
            b_int[i] = 255 - b_int[i]
        for i in range(len(b_int) - 1, -1, -1):
            if b_int[i] == 255:
                b_int[i - 1] += 1
                b_int[i] = 0
                break
            else:
                b_int[i] += 1
                break
    int_value = 0
    for val in b_int:
        int_value = int_value << 8 | val
    if is_negative:
        int_value *= -1
    return int_value
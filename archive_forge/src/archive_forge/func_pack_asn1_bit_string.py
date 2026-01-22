import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def pack_asn1_bit_string(value: bytes, tag: bool=True) -> bytes:
    b_data = b'\x00' + value
    if tag:
        b_data = pack_asn1(TagClass.universal, False, TypeTagNumber.bit_string, b_data)
    return b_data
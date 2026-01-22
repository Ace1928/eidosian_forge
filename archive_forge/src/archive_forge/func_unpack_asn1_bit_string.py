import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def unpack_asn1_bit_string(value: typing.Union[ASN1Value, bytes]) -> bytes:
    """Unpacks an ASN.1 BIT STRING value."""
    b_data = extract_asn1_tlv(value, TagClass.universal, TypeTagNumber.bit_string)
    unused_bits = struct.unpack('B', b_data[:1])[0]
    last_octet = struct.unpack('B', b_data[-2:-1])[0]
    last_octet = last_octet >> unused_bits << unused_bits
    return b_data[1:-1] + struct.pack('B', last_octet)
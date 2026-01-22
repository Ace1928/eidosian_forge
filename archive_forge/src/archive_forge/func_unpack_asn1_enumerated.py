import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def unpack_asn1_enumerated(value: typing.Union[ASN1Value, bytes]) -> int:
    """Unpacks an ASN.1 ENUMERATED value."""
    b_data = extract_asn1_tlv(value, TagClass.universal, TypeTagNumber.enumerated)
    return unpack_asn1_integer(b_data)
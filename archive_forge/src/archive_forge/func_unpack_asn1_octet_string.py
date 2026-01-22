import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def unpack_asn1_octet_string(value: typing.Union[ASN1Value, bytes]) -> bytes:
    """Unpacks an ASN.1 OCTET STRING value."""
    return extract_asn1_tlv(value, TagClass.universal, TypeTagNumber.octet_string)
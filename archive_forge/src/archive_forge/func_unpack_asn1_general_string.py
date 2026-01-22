import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def unpack_asn1_general_string(value: typing.Union[ASN1Value, bytes]) -> bytes:
    """Unpacks an ASN.1 GeneralString value."""
    return extract_asn1_tlv(value, TagClass.universal, TypeTagNumber.general_string)
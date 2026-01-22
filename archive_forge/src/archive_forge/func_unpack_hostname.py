import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def unpack_hostname(value: typing.Union[bytes, ASN1Value]) -> 'HostAddress':
    """Unpacks an ASN.1 value to a HostAddress."""
    s = unpack_asn1_tagged_sequence(value)
    name_type = KerberosHostAddressType(get_sequence_value(s, 0, 'HostAddress', 'addr-type', unpack_asn1_integer))
    name = get_sequence_value(s, 1, 'HostAddress', 'address', unpack_asn1_octet_string)
    return HostAddress(name_type, name)
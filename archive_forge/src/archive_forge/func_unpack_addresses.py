import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def unpack_addresses(value: typing.Union[ASN1Value, bytes]) -> typing.List[HostAddress]:
    return [unpack_hostname(h) for h in unpack_asn1_sequence(value)]
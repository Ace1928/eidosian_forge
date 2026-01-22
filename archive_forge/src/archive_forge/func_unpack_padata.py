import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def unpack_padata(value: typing.Union[ASN1Value, bytes]) -> typing.List:
    return [PAData.unpack(p) for p in unpack_asn1_sequence(value)]
import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def parse_host_address(value: HostAddress) -> typing.Dict[str, typing.Any]:
    return {'addr-type': parse_enum(value.addr_type), 'address': parse_text(value.value)}
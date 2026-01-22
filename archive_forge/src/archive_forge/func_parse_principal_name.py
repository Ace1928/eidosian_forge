import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def parse_principal_name(value: PrincipalName) -> typing.Dict[str, typing.Any]:
    return {'name-type': parse_enum(value.name_type), 'name-string': [parse_text(v) for v in value.value]}
import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def parse_enum(value: typing.Union[int, str, enum.Enum], enum_type: typing.Optional[typing.Type]=None) -> str:
    """Parses an IntEnum into a human representative object of that enum."""
    enum_name = 'UNKNOWN'
    if isinstance(value, enum.Enum):
        enum_name = value.name
    labels = _enum_labels(value, enum_type)
    value = value.value if isinstance(value, enum.Enum) else value
    for v, name in labels.items():
        if value == v:
            enum_name = name
            break
    return '%s (%s)' % (enum_name, value)
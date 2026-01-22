from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
def parse_header_fields(buf, endianness):
    l, pos = _header_fields_type.parse_data(buf, 12, endianness)
    return ({HeaderFields(k): v[1] for k, v in l}, pos)
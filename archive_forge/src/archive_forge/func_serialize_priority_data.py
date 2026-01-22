import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
def serialize_priority_data(self):
    return _STRUCT_LB.pack(self.depends_on + (2147483648 if self.exclusive else 0), self.stream_weight)
import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
def parse_priority_data(self, data):
    try:
        self.depends_on, self.stream_weight = _STRUCT_LB.unpack(data[:5])
    except struct.error:
        raise InvalidFrameError('Invalid Priority data')
    self.exclusive = True if self.depends_on >> 31 else False
    self.depends_on &= 2147483647
    return 5
from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
def read_uint32(self):
    result = struct.unpack_from('<I', self._data, self._position)[0]
    self._position += 4
    return result
from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
def is_load_local_packet(self):
    return self._data[0] == 251
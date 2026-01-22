import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
@property
def total_padding(self):
    return self.pad_length
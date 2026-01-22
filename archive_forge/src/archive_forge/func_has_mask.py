import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
def has_mask(self):
    if not self.header:
        return False
    return self.header[frame_buffer._HEADER_MASK_INDEX]
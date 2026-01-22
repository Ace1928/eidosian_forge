import struct
import sys
import numpy as np
def str64(s):
    """Convert a string to an int64."""
    s = s + '\x00' * (8 - len(s))
    s = s.encode('ascii')
    return struct.unpack('@q', s)[0]
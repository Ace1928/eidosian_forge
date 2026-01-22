import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
def float32_to_uint32(n):
    packed = struct.pack('<1f', n)
    converted, = struct.unpack('<1L', packed)
    return converted
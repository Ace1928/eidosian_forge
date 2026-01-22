import array
import struct
from . import errors
from .io import gfile
def u32(x):
    return x & 4294967295
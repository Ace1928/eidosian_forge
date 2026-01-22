import struct
from io import BytesIO
from functools import wraps
import warnings
def unpack_uhyper(self):
    hi = self.unpack_uint()
    lo = self.unpack_uint()
    return int(hi) << 32 | lo
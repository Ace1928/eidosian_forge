import struct
from io import BytesIO
from functools import wraps
import warnings
def unpack_string(self):
    n = self.unpack_uint()
    return self.unpack_fstring(n)
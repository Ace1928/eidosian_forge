import struct
from io import BytesIO
from functools import wraps
import warnings
def unpack_bool(self):
    return bool(self.unpack_int())
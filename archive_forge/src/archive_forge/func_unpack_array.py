import struct
from io import BytesIO
from functools import wraps
import warnings
def unpack_array(self, unpack_item):
    n = self.unpack_uint()
    return self.unpack_farray(n, unpack_item)
import struct
from io import BytesIO
from functools import wraps
import warnings
def pack_string(self, s):
    n = len(s)
    self.pack_uint(n)
    self.pack_fstring(n, s)
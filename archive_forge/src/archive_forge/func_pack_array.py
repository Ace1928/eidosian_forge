import struct
from io import BytesIO
from functools import wraps
import warnings
def pack_array(self, list, pack_item):
    n = len(list)
    self.pack_uint(n)
    self.pack_farray(n, list, pack_item)
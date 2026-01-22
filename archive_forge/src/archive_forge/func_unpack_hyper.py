import struct
from io import BytesIO
from functools import wraps
import warnings
def unpack_hyper(self):
    x = self.unpack_uhyper()
    if x >= 9223372036854775808:
        x = x - 18446744073709551616
    return x
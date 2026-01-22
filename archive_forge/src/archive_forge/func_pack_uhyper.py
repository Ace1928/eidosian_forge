import struct
from io import BytesIO
from functools import wraps
import warnings
def pack_uhyper(self, x):
    try:
        self.pack_uint(x >> 32 & 4294967295)
    except (TypeError, struct.error) as e:
        raise ConversionError(e.args[0]) from None
    try:
        self.pack_uint(x & 4294967295)
    except (TypeError, struct.error) as e:
        raise ConversionError(e.args[0]) from None
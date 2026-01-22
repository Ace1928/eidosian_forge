import struct
from io import BytesIO
from functools import wraps
import warnings
@raise_conversion_error
def pack_int(self, x):
    self.__buf.write(struct.pack('>l', x))
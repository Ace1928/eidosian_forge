import struct
from io import BytesIO
from functools import wraps
import warnings
@raise_conversion_error
def pack_double(self, x):
    self.__buf.write(struct.pack('>d', x))
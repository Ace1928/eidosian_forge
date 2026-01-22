import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
def signed(to_type, n):
    formats = {'byte': 'Bb', 'short': 'Hh', 'long': 'Ll', 'ubyte': 'bB', 'ushort': 'hH', 'ulong': 'lL'}
    try:
        pack_format, unpack_format = formats[to_type]
    except KeyError as ke:
        raise ValueError(f'invalid integer type {to_type}') from ke
    try:
        packed = struct.pack(pack_format, n)
        return struct.unpack(unpack_format, packed)[0]
    except struct.error as err:
        raise ValueError(*err.args) from err
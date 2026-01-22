import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
@_ffi.callback('sf_vio_write')
def vio_write(ptr, count, user_data):
    buf = _ffi.buffer(ptr, count)
    data = buf[:]
    written = file.write(data)
    if written is None:
        written = count
    return written
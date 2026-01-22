import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
@_ffi.callback('sf_vio_read')
def vio_read(ptr, count, user_data):
    try:
        buf = _ffi.buffer(ptr, count)
        data_read = file.readinto(buf)
    except AttributeError:
        data = file.read(count)
        data_read = len(data)
        buf = _ffi.buffer(ptr, data_read)
        buf[0:data_read] = data
    return data_read
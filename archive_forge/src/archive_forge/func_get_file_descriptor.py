from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
@ensure_connected
def get_file_descriptor(self):
    return lib.xcb_get_file_descriptor(self._conn)
from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
@ensure_connected
def discard_reply(self, sequence):
    return lib.xcb_discard_reply(self._conn, sequence)
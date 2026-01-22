import libevdev
import os
import ctypes
import errno
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_uint
from ctypes import c_void_p
from ctypes import c_long
from ctypes import c_int32
from ctypes import c_uint16
@property
def current_slot(self):
    """
        :return: the current of slots on this device or ``None`` if this device
                 does not support slots

        :note: Read-only
        """
    s = self._get_current_slot(self._ctx)
    return s if s >= 0 else None
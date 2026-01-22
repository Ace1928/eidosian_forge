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
def syspath(self):
    """
        Return a string with the /dev/input/eventX device node
        """
    syspath = self._uinput_get_syspath(self._uinput_device)
    return syspath.decode('iso8859-1')
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
def enable_property(self, prop):
    """
        :param prop: the property as integer or string
        """
    if not isinstance(prop, int):
        prop = self.property_to_value(prop)
    self._enable_property(self._ctx, prop)
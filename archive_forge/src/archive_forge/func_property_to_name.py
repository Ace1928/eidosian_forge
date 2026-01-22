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
@classmethod
def property_to_name(cls, prop):
    """
        :param prop: the numerical property value
        :return: A string with the property name or ``None``

        This function is the equivalent to ``libevdev_property_get_name()``
        """
    name = cls._property_get_name(prop)
    if not name:
        return None
    return name.decode('iso8859-1')
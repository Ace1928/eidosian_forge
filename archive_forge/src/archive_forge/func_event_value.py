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
def event_value(self, event_type, event_code, new_value=None):
    """
        :param event_type: the event type, either as integer or as string
        :param event_code: the event code, either as integer or as string
        :param new_value: optional, the value to set to
        :return: the current value of type + code, or ``None`` if it doesn't
                 exist on this device
        """
    t, c = self._code(event_type, event_code)
    if not self.has_event(t, c):
        return None
    if new_value is not None:
        self._set_event_value(self._ctx, t, c, new_value)
    v = self._get_event_value(self._ctx, t, c)
    return v
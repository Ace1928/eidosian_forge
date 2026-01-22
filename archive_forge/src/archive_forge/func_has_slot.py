import os
import time
from ctypes import cdll, Structure, c_ulong, c_int, c_ushort, \
def has_slot(self):
    """Return True if the device has slot information.
        """
    if self._fd == -1:
        raise Exception('Device closed')
    return bool(self._device.caps.has_slot)
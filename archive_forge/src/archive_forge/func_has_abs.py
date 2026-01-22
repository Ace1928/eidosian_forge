import os
import time
from ctypes import cdll, Structure, c_ulong, c_int, c_ushort, \
def has_abs(self, index):
    """Return True if the device has abs data.

        :Parameters:
            `index`: int
                One of const starting with a name ABS_MT_
        """
    if self._fd == -1:
        raise Exception('Device closed')
    if index < 0 or index >= MTDEV_ABS_SIZE:
        raise IndexError('Invalid index')
    return bool(self._device.caps.has_abs[index])
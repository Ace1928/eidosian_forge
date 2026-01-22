from pyudev._ctypeslib.libudev import ERROR_CHECKERS, SIGNATURES
from pyudev._ctypeslib.utils import load_ctypes_library
from pyudev._errors import DeviceNotFoundAtPathError
from pyudev._util import (
from pyudev.device import Devices
@log_priority.setter
def log_priority(self, value):
    """
        Set the log priority.

        :param int value: the log priority.
        """
    self._libudev.udev_set_log_priority(self, value)
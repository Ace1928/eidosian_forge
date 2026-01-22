import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@property
def subsystem(self):
    """
        Name of the subsystem this device is part of as unicode string.

        :returns: name of subsystem if found, else None
        :rtype: unicode string or NoneType
        """
    subsys = self._libudev.udev_device_get_subsystem(self)
    return None if subsys is None else ensure_unicode_string(subsys)
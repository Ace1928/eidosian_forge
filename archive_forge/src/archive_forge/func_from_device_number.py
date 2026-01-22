import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@classmethod
def from_device_number(cls, context, typ, number):
    """
        .. versionadded:: 0.11
        .. deprecated:: 0.18
           Use :class:`Devices.from_device_number` instead.
        """
    import warnings
    warnings.warn('Will be removed in 1.0. Use equivalent Devices method instead.', DeprecationWarning, stacklevel=2)
    return Devices.from_device_number(context, typ, number)
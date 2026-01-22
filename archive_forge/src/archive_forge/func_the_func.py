import abc
import functools
import os
import re
from pyudev._errors import DeviceNotFoundError
from pyudev.device import Devices
@functools.wraps(func)
def the_func(*args, **kwargs):
    """
        Returns result of calling ``func`` on ``args``, ``kwargs``.
        Returns None if ``func`` raises :exc:`DeviceNotFoundError`.
        """
    try:
        return func(*args, **kwargs)
    except DeviceNotFoundError:
        return None
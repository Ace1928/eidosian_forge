import ctypes
import os
import re
import threading
from collections.abc import Iterable
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import pathutils
from os_win.utils import win32utils
from os_win.utils.winapi import libs as w_lib
def get_disk_capacity(self, path, ignore_errors=False):
    """Returns total/free space for a given directory."""
    norm_path = os.path.abspath(path)
    total_bytes = ctypes.c_ulonglong(0)
    free_bytes = ctypes.c_ulonglong(0)
    try:
        self._win32_utils.run_and_check_output(kernel32.GetDiskFreeSpaceExW, ctypes.c_wchar_p(norm_path), None, ctypes.pointer(total_bytes), ctypes.pointer(free_bytes), kernel32_lib_func=True)
        return (total_bytes.value, free_bytes.value)
    except exceptions.Win32Exception as exc:
        LOG.error('Could not get disk %(path)s capacity info. Exception: %(exc)s', dict(path=path, exc=exc))
        if ignore_errors:
            return (0, 0)
        else:
            raise exc
import ctypes
import struct
from eventlet import patcher
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
def cancel_io(self, handle, overlapped_structure=None, ignore_invalid_handle=False):
    """Cancels pending IO on specified handle.

        If an overlapped structure is passed, only the IO requests that
        were issued with the specified overlapped structure are canceled.
        """
    ignored_error_codes = [w_const.ERROR_NOT_FOUND]
    if ignore_invalid_handle:
        ignored_error_codes.append(w_const.ERROR_INVALID_HANDLE)
    lp_overlapped = ctypes.byref(overlapped_structure) if overlapped_structure else None
    self._run_and_check_output(kernel32.CancelIoEx, handle, lp_overlapped, ignored_error_codes=ignored_error_codes)
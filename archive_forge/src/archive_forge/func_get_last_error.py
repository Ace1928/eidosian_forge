import ctypes
from oslo_log import log as logging
from os_win import _utils
from os_win import exceptions
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
def get_last_error(self):
    error_code = kernel32.GetLastError()
    kernel32.SetLastError(0)
    return error_code
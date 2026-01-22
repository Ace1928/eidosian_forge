import ctypes
from oslo_log import log as logging
from os_win import _utils
from os_win import exceptions
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
def wait_for_single_object(self, handle, milliseconds=w_const.INFINITE):
    ret_val = self.run_and_check_output(kernel32.WaitForSingleObject, handle, milliseconds, kernel32_lib_func=True, error_ret_vals=[w_const.WAIT_FAILED])
    if ret_val == w_const.ERROR_WAIT_TIMEOUT:
        raise exceptions.Timeout()
    return ret_val
from unittest import mock
import ddt
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_run_and_check_output_explicit_error_ret_vals(self):
    ret_val = 1
    error_ret_vals = [ret_val]
    mock_get_last_err, mock_get_err_msg = self._test_run_and_check_output(ret_val=ret_val, error_ret_vals=error_ret_vals, ret_val_is_err_code=False, expected_exc=exceptions.Win32Exception)
    mock_get_err_msg.assert_called_once_with(win32utils.ctypes.c_ulong(mock_get_last_err).value)
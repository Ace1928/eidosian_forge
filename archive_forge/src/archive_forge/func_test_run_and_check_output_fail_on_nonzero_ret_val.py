from unittest import mock
import ddt
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_run_and_check_output_fail_on_nonzero_ret_val(self):
    ret_val = 1
    mock_get_last_err, mock_get_err_msg = self._test_run_and_check_output(ret_val=ret_val, expected_exc=exceptions.VHDWin32APIException, failure_exc=exceptions.VHDWin32APIException)
    mock_get_err_msg.assert_called_once_with(ret_val)
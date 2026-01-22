from unittest import mock
import ddt
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_run_and_check_output_with_err_msg_dict(self):
    self._ctypes_patcher.stop()
    err_code = 1
    err_msg = 'fake_err_msg'
    err_msg_dict = {err_code: err_msg}
    mock_func = mock.Mock()
    mock_func.return_value = err_code
    try:
        self._win32_utils.run_and_check_output(mock_func, mock.sentinel.arg, error_msg_src=err_msg_dict)
    except Exception as ex:
        self.assertIsInstance(ex, exceptions.Win32Exception)
        self.assertIn(err_msg, ex.message)
from unittest import mock
import ddt
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_get_error_message(self):
    err_msg = self._win32_utils.get_error_message(mock.sentinel.err_code)
    fake_msg_buff = win32utils.ctypes.c_char_p.return_value
    expected_flags = w_const.FORMAT_MESSAGE_FROM_SYSTEM | w_const.FORMAT_MESSAGE_ALLOCATE_BUFFER | w_const.FORMAT_MESSAGE_IGNORE_INSERTS
    win32utils.kernel32.FormatMessageA.assert_called_once_with(expected_flags, None, mock.sentinel.err_code, 0, win32utils.ctypes.byref(fake_msg_buff), 0, None)
    self.assertEqual(fake_msg_buff.value, err_msg)
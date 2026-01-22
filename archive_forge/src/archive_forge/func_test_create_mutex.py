from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import processutils
from os_win.utils.winapi import constants as w_const
def test_create_mutex(self):
    handle = self._procutils.create_mutex(mock.sentinel.name, mock.sentinel.owner, mock.sentinel.sec_attr)
    self.assertEqual(self._mock_run.return_value, handle)
    self._mock_run.assert_called_once_with(self._mock_kernel32.CreateMutexW, self._ctypes.byref(mock.sentinel.sec_attr), mock.sentinel.owner, mock.sentinel.name, kernel32_lib_func=True)
from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import processutils
from os_win.utils.winapi import constants as w_const
def test_set_information_job_object(self):
    self._procutils.set_information_job_object(mock.sentinel.job_handle, mock.sentinel.job_info_class, mock.sentinel.job_info)
    self._mock_run.assert_called_once_with(self._mock_kernel32.SetInformationJobObject, mock.sentinel.job_handle, mock.sentinel.job_info_class, self._ctypes.byref(mock.sentinel.job_info), self._ctypes.sizeof(mock.sentinel.job_info), kernel32_lib_func=True)
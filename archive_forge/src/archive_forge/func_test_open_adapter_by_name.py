import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
def test_open_adapter_by_name(self):
    self._ctypes_mocker.stop()
    self._mock_run.return_value = mock.sentinel.handle
    resulted_handle = self._fc_utils._open_adapter_by_name(self._FAKE_ADAPTER_NAME)
    args_list = self._mock_run.call_args_list[0][0]
    self.assertEqual(fc_utils.hbaapi.HBA_OpenAdapter, args_list[0])
    self.assertEqual(six.b(self._FAKE_ADAPTER_NAME), args_list[1].value)
    self.assertEqual(mock.sentinel.handle, resulted_handle)
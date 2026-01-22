import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
@mock.patch.object(fc_utils.fc_struct, 'HBA_HANDLE')
def test_open_adapter_by_wwn(self, mock_hba_handle_struct):
    exp_handle = mock_hba_handle_struct.return_value
    resulted_handle = self._fc_utils._open_adapter_by_wwn(mock.sentinel.wwn)
    self.assertEqual(exp_handle, resulted_handle)
    self._mock_run.assert_called_once_with(fc_utils.hbaapi.HBA_OpenAdapterByWWN, self._ctypes.byref(exp_handle), mock.sentinel.wwn)
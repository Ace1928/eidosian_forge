import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
@mock.patch.object(fc_struct, 'HBA_AdapterAttributes')
def test_get_adapter_attributes(self, mock_class_HBA_AdapterAttributes):
    resulted_hba_attributes = self._fc_utils._get_adapter_attributes(mock.sentinel.hba_handle)
    self._mock_run.assert_called_once_with(fc_utils.hbaapi.HBA_GetAdapterAttributes, mock.sentinel.hba_handle, self._ctypes.byref(mock_class_HBA_AdapterAttributes.return_value))
    self.assertEqual(mock_class_HBA_AdapterAttributes.return_value, resulted_hba_attributes)
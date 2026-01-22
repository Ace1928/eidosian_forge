import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
@mock.patch.object(fc_struct, 'HBA_PortAttributes')
def test_get_adapter_port_attributes(self, mock_class_HBA_PortAttributes):
    resulted_port_attributes = self._fc_utils._get_adapter_port_attributes(mock.sentinel.hba_handle, mock.sentinel.port_index)
    self._mock_run.assert_called_once_with(fc_utils.hbaapi.HBA_GetAdapterPortAttributes, mock.sentinel.hba_handle, mock.sentinel.port_index, self._ctypes.byref(mock_class_HBA_PortAttributes.return_value))
    self.assertEqual(mock_class_HBA_PortAttributes.return_value, resulted_port_attributes)
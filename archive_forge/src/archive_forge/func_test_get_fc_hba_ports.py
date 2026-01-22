import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
@mock.patch.object(fc_utils.FCUtils, '_get_fc_hba_adapter_ports')
@mock.patch.object(fc_utils.FCUtils, '_get_adapter_name')
@mock.patch.object(fc_utils.FCUtils, 'get_fc_hba_count')
def test_get_fc_hba_ports(self, mock_get_fc_hba_count, mock_get_adapter_name, mock_get_adapter_ports):
    fake_adapter_count = 3
    mock_get_adapter_name.side_effect = [Exception, mock.sentinel.adapter_name, mock.sentinel.adapter_name]
    mock_get_fc_hba_count.return_value = fake_adapter_count
    mock_get_adapter_ports.side_effect = [Exception, [mock.sentinel.port]]
    expected_hba_ports = [mock.sentinel.port]
    resulted_hba_ports = self._fc_utils.get_fc_hba_ports()
    self.assertEqual(expected_hba_ports, resulted_hba_ports)
    self.assertEqual(expected_hba_ports, resulted_hba_ports)
    mock_get_adapter_name.assert_has_calls([mock.call(index) for index in range(fake_adapter_count)])
    mock_get_adapter_ports.assert_has_calls([mock.call(mock.sentinel.adapter_name)] * 2)
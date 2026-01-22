import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
@mock.patch.object(fc_utils.FCUtils, '_open_adapter_by_name')
@mock.patch.object(fc_utils.FCUtils, '_close_adapter')
@mock.patch.object(fc_utils.FCUtils, '_get_adapter_port_attributes')
@mock.patch.object(fc_utils.FCUtils, '_get_adapter_attributes')
def test_get_fc_hba_adapter_ports(self, mock_get_adapter_attributes, mock_get_adapter_port_attributes, mock_close_adapter, mock_open_adapter):
    fake_port_count = 1
    fake_port_index = 0
    fake_node_wwn = list(range(3))
    fake_port_wwn = list(range(3))
    mock_adapter_attributes = mock.MagicMock()
    mock_adapter_attributes.NumberOfPorts = fake_port_count
    mock_port_attributes = mock.MagicMock()
    mock_port_attributes.NodeWWN.wwn = fake_node_wwn
    mock_port_attributes.PortWWN.wwn = fake_port_wwn
    mock_get_adapter_attributes.return_value = mock_adapter_attributes
    mock_get_adapter_port_attributes.return_value = mock_port_attributes
    resulted_hba_ports = self._fc_utils._get_fc_hba_adapter_ports(mock.sentinel.adapter_name)
    expected_hba_ports = [{'node_name': _utils.byte_array_to_hex_str(fake_node_wwn), 'port_name': _utils.byte_array_to_hex_str(fake_port_wwn)}]
    self.assertEqual(expected_hba_ports, resulted_hba_ports)
    mock_open_adapter.assert_called_once_with(mock.sentinel.adapter_name)
    mock_close_adapter.assert_called_once_with(mock_open_adapter(mock.sentinel.adapter_nam))
    mock_get_adapter_attributes.assert_called_once_with(mock_open_adapter.return_value)
    mock_get_adapter_port_attributes.assert_called_once_with(mock_open_adapter.return_value, fake_port_index)
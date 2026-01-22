from unittest import mock
import ddt
from os_win import exceptions as os_win_exc
from os_brick import exception
from os_brick.initiator.windows import fibre_channel as fc
from os_brick.tests.windows import test_base
@ddt.data(True, False)
@mock.patch.object(fc.utilsfactory, 'get_fc_utils')
def test_get_volume_connector_props(self, valid_fc_hba_ports, mock_get_fc_utils):
    fake_fc_hba_ports = [{'node_name': mock.sentinel.node_name, 'port_name': mock.sentinel.port_name}, {'node_name': mock.sentinel.second_node_name, 'port_name': mock.sentinel.second_port_name}]
    self._fc_utils = mock_get_fc_utils.return_value
    self._fc_utils.get_fc_hba_ports.return_value = fake_fc_hba_ports if valid_fc_hba_ports else []
    props = self._connector.get_connector_properties()
    self._fc_utils.refresh_hba_configuration.assert_called_once_with()
    self._fc_utils.get_fc_hba_ports.assert_called_once_with()
    if valid_fc_hba_ports:
        expected_props = {'wwpns': [mock.sentinel.port_name, mock.sentinel.second_port_name], 'wwnns': [mock.sentinel.node_name, mock.sentinel.second_node_name]}
    else:
        expected_props = {}
    self.assertCountEqual(expected_props, props)
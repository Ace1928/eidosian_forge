from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_remove_switch_port(self):
    mock_sw_port = self._mock_get_switch_port_alloc()
    self.netutils._switch_ports[self._FAKE_PORT_NAME] = mock_sw_port
    self.netutils._vlan_sds[mock_sw_port.InstanceID] = mock.MagicMock()
    self.netutils._jobutils.remove_virt_resource.side_effect = exceptions.x_wmi
    self.netutils.remove_switch_port(self._FAKE_PORT_NAME, False)
    self.netutils._jobutils.remove_virt_resource.assert_called_once_with(mock_sw_port)
    self.assertNotIn(self._FAKE_PORT_NAME, self.netutils._switch_ports)
    self.assertNotIn(mock_sw_port.InstanceID, self.netutils._vlan_sds)
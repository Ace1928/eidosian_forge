from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_get_setting_data')
def test_get_switch_port_allocation_expected(self, mock_get_set_data):
    self.netutils._switch_ports = {}
    mock_get_set_data.return_value = (None, False)
    self.assertRaises(exceptions.HyperVPortNotFoundException, self.netutils._get_switch_port_allocation, mock.sentinel.port_name, expected=True)
    mock_get_set_data.assert_called_once_with(self.netutils._PORT_ALLOC_SET_DATA, mock.sentinel.port_name, False)
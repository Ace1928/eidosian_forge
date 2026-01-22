from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_get_vswitch_external_port')
def test_get_vswitch_external_network_name(self, mock_get_vswitch_port):
    mock_get_vswitch_port.return_value.ElementName = mock.sentinel.network_name
    result = self.netutils.get_vswitch_external_network_name(mock.sentinel.vswitch_name)
    self.assertEqual(mock.sentinel.network_name, result)
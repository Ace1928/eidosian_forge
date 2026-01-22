from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_get_vswitch')
def test_get_vswitch_extensions(self, mock_get_vswitch):
    mock_vswitch = mock_get_vswitch.return_value
    mock_ext = mock.Mock()
    ext_cls = self.netutils._conn.Msvm_EthernetSwitchExtension
    ext_cls.return_value = [mock_ext] * 2
    extensions = self.netutils.get_vswitch_extensions(mock.sentinel.vswitch_name)
    exp_extensions = [{'name': mock_ext.ElementName, 'version': mock_ext.Version, 'vendor': mock_ext.Vendor, 'description': mock_ext.Description, 'enabled_state': mock_ext.EnabledState, 'extension_type': mock_ext.ExtensionType}] * 2
    self.assertEqual(exp_extensions, extensions)
    mock_get_vswitch.assert_called_once_with(mock.sentinel.vswitch_name)
    ext_cls.assert_called_once_with(SystemName=mock_vswitch.Name)
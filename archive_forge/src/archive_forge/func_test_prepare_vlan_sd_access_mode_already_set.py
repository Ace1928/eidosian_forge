from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_prepare_vlan_sd_access_mode_already_set(self):
    mock_vlan_sd = mock.MagicMock(OperationMode=constants.VLAN_MODE_ACCESS, AccessVlanId=mock.sentinel.vlan_id)
    actual_vlan_sd = self.netutils._prepare_vlan_sd_access_mode(mock_vlan_sd, mock.sentinel.vlan_id)
    self.assertIsNone(actual_vlan_sd)
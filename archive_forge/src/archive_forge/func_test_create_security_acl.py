from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtilsR2, '_create_default_setting_data')
def test_create_security_acl(self, mock_create_default_setting_data):
    sg_rule = mock.MagicMock()
    sg_rule.to_dict.return_value = {}
    acl = self.netutils._create_security_acl(sg_rule, mock.sentinel.weight)
    self.assertEqual(mock.sentinel.weight, acl.Weight)
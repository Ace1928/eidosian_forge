from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_get_bandwidth_setting_data_from_port_alloc')
@mock.patch.object(networkutils.NetworkUtils, '_get_default_setting_data')
def test_set_port_qos_rule_invalid_qos_rule_exc(self, mock_get_default_sd, mock_get_bandwidth_sd):
    self._mock_get_switch_port_alloc()
    mock_add_feature = self.netutils._jobutils.add_virt_feature
    mock_add_feature.side_effect = exceptions.InvalidParameterValue('0x80070057')
    qos_rule = dict(min_kbps=20000, max_kbps=30000, max_burst_kbps=40000, max_burst_size_kb=50000)
    self.assertRaises(exceptions.InvalidParameterValue, self.netutils.set_port_qos_rule, mock.sentinel.port_id, qos_rule)
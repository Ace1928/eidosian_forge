from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_setting_data_from_port_alloc_cached(self):
    mock_port = mock.MagicMock(InstanceID=mock.sentinel.InstanceID)
    cache = {mock_port.InstanceID: mock.sentinel.sd_object}
    result = self.netutils._get_setting_data_from_port_alloc(mock_port, cache, mock.sentinel.data_class)
    self.assertEqual(mock.sentinel.sd_object, result)
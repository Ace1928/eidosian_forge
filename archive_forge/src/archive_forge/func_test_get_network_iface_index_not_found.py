from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
def test_get_network_iface_index_not_found(self):
    self.utils._scimv2.MSFT_NetAdapter.return_value = []
    self.assertRaises(exceptions.NotFound, self.utils._get_network_iface_index, mock.sentinel.network_name)
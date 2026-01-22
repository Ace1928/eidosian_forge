from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
def test_get_network_iface_index_cached(self):
    self.utils._net_if_indexes[mock.sentinel.fake_network] = mock.sentinel.iface_index
    index = self.utils._get_network_iface_index(mock.sentinel.fake_network)
    self.assertEqual(mock.sentinel.iface_index, index)
    self.assertFalse(self.utils._scimv2.MSFT_NetAdapter.called)
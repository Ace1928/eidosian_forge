import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_get_network_by_id(self):
    network_id = 'test-net-id'
    network_name = 'network'
    network = {'id': network_id, 'name': network_name}
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', network_id]), json={'network': network})])
    self.assertTrue(self.cloud.get_network_by_id(network_id))
    self.assert_calls()
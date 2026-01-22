import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_list_networks(self):
    net1 = {'id': '1', 'name': 'net1'}
    net2 = {'id': '2', 'name': 'net2'}
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': [net1, net2]})])
    nets = self.cloud.list_networks()
    self.assertEqual([_network.Network(**i).to_dict(computed=False) for i in [net1, net2]], [i.to_dict(computed=False) for i in nets])
    self.assert_calls()
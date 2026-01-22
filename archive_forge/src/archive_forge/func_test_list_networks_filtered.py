import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_list_networks_filtered(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks'], qs_elements=['name=test']), json={'networks': []})])
    self.cloud.list_networks(filters={'name': 'test'})
    self.assert_calls()
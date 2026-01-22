import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_create_network_provider(self):
    provider_opts = {'physical_network': 'mynet', 'network_type': 'vlan', 'segmentation_id': 'vlan1'}
    new_network_provider_opts = {'provider:physical_network': 'mynet', 'provider:network_type': 'vlan', 'provider:segmentation_id': 'vlan1'}
    mock_new_network_rep = copy.copy(self.mock_new_network_rep)
    mock_new_network_rep.update(new_network_provider_opts)
    expected_send_params = {'admin_state_up': True, 'name': 'netname'}
    expected_send_params.update(new_network_provider_opts)
    self.register_uris([dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'network': mock_new_network_rep}, validate=dict(json={'network': expected_send_params}))])
    network = self.cloud.create_network('netname', provider=provider_opts)
    self._compare_networks(mock_new_network_rep, network)
    self.assert_calls()
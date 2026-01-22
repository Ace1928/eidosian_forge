import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_create_network_port_security_disabled(self):
    port_security_state = False
    mock_new_network_rep = copy.copy(self.mock_new_network_rep)
    mock_new_network_rep['port_security_enabled'] = port_security_state
    self.register_uris([dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'network': mock_new_network_rep}, validate=dict(json={'network': {'admin_state_up': True, 'name': 'netname', 'port_security_enabled': port_security_state}}))])
    network = self.cloud.create_network('netname', port_security_enabled=port_security_state)
    self._compare_networks(mock_new_network_rep, network)
    self.assert_calls()
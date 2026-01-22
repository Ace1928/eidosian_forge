import copy
import datetime
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack import utils
def test_neutron_available_floating_ips_network(self):
    """
        Test with specifying a network name.
        """
    fips_mock_uri = 'https://network.example.com/v2.0/floatingips'
    self.register_uris([dict(method='GET', uri='https://network.example.com/v2.0/networks', json={'networks': [self.mock_get_network_rep]}), dict(method='GET', uri='https://network.example.com/v2.0/subnets', json={'subnets': []}), dict(method='GET', uri=fips_mock_uri, json={'floatingips': []}), dict(method='POST', uri=fips_mock_uri, json=self.mock_floating_ip_new_rep, validate=dict(json={'floatingip': {'floating_network_id': self.mock_get_network_rep['id']}}))])
    self.cloud._neutron_available_floating_ips(network=self.mock_get_network_rep['name'])
    self.assert_calls()
import copy
import datetime
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack import utils
def test_neutron_available_floating_ips_invalid_network(self):
    """
        Test with an invalid network name.
        """
    self.register_uris([dict(method='GET', uri='https://network.example.com/v2.0/networks', json={'networks': [self.mock_get_network_rep]}), dict(method='GET', uri='https://network.example.com/v2.0/subnets', json={'subnets': []})])
    self.assertRaises(exceptions.SDKException, self.cloud._neutron_available_floating_ips, network='INVALID')
    self.assert_calls()
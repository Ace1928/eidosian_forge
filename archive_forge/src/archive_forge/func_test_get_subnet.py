import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_get_subnet(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets', self.subnet_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets'], qs_elements=['name=%s' % self.subnet_name]), json={'subnets': [self.mock_subnet_rep]})])
    r = self.cloud.get_subnet(self.subnet_name)
    self.assertIsNotNone(r)
    self._compare_subnets(self.mock_subnet_rep, r)
    self.assert_calls()
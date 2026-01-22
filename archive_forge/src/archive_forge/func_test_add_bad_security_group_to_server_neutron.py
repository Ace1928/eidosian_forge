import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_add_bad_security_group_to_server_neutron(self):
    fake_server = fakes.make_fake_server('1234', 'server-name', 'ACTIVE')
    self.cloud.secgroup_source = 'neutron'
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': [fake_server]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups']), json={'security_groups': [neutron_grp_dict]})])
    self.assertFalse(self.cloud.add_server_security_groups('server-name', 'unknown-sec-group'))
    self.assert_calls()
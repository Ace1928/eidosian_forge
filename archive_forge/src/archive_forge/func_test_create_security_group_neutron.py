import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_security_group_neutron(self):
    self.cloud.secgroup_source = 'neutron'
    group_name = self.getUniqueString()
    group_desc = self.getUniqueString('description')
    new_group = fakes.make_fake_neutron_security_group(id='2', name=group_name, description=group_desc, rules=[])
    self.register_uris([dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups']), json={'security_group': new_group}, validate=dict(json={'security_group': {'name': group_name, 'description': group_desc}}))])
    r = self.cloud.create_security_group(group_name, group_desc)
    self.assertEqual(group_name, r['name'])
    self.assertEqual(group_desc, r['description'])
    self.assertEqual(True, r['stateful'])
    self.assert_calls()
import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_security_group_nova(self):
    group_name = self.getUniqueString()
    self.has_neutron = False
    group_desc = self.getUniqueString('description')
    new_group = fakes.make_fake_nova_security_group(id='2', name=group_name, description=group_desc, rules=[])
    self.register_uris([dict(method='POST', uri='{endpoint}/os-security-groups'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'security_group': new_group}, validate=dict(json={'security_group': {'name': group_name, 'description': group_desc}}))])
    self.cloud.secgroup_source = 'nova'
    r = self.cloud.create_security_group(group_name, group_desc)
    self.assertEqual(group_name, r['name'])
    self.assertEqual(group_desc, r['description'])
    self.assert_calls()
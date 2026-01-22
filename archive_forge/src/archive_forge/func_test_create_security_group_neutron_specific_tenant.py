import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_security_group_neutron_specific_tenant(self):
    self.cloud.secgroup_source = 'neutron'
    project_id = '861808a93da0484ea1767967c4df8a23'
    group_name = self.getUniqueString()
    group_desc = 'security group from test_create_security_group_neutron_specific_tenant'
    new_group = fakes.make_fake_neutron_security_group(id='2', name=group_name, description=group_desc, project_id=project_id, rules=[])
    self.register_uris([dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups']), json={'security_group': new_group}, validate=dict(json={'security_group': {'name': group_name, 'description': group_desc, 'tenant_id': project_id}}))])
    r = self.cloud.create_security_group(group_name, group_desc, project_id)
    self.assertEqual(group_name, r['name'])
    self.assertEqual(group_desc, r['description'])
    self.assertEqual(project_id, r['tenant_id'])
    self.assert_calls()
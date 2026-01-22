import random
import string
from openstack import exceptions
from openstack.tests.functional import base
def test_grant_revoke_role_group_system(self):
    if self.identity_version in ('2', '2.0'):
        self.skipTest('Identity service does not support system or group')
    role_name = self.role_prefix + '_grant_group_system'
    role = self.operator_cloud.create_role(role_name)
    group_name = self.group_prefix + '_group_system'
    group = self.operator_cloud.create_group(name=group_name, description='test group')
    self.assertTrue(self.operator_cloud.grant_role(role_name, group=group['id'], system='all'))
    assignments = self.operator_cloud.list_role_assignments({'role': role['id'], 'group': group['id'], 'system': 'all'})
    self.assertIsInstance(assignments, list)
    self.assertEqual(1, len(assignments))
    self.assertTrue(self.operator_cloud.revoke_role(role_name, group=group['id'], system='all'))
    assignments = self.operator_cloud.list_role_assignments({'role': role['id'], 'group': group['id'], 'system': 'all'})
    self.assertIsInstance(assignments, list)
    self.assertEqual(0, len(assignments))
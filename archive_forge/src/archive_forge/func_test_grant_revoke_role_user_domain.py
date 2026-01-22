import random
import string
from openstack import exceptions
from openstack.tests.functional import base
def test_grant_revoke_role_user_domain(self):
    if self.identity_version in ('2', '2.0'):
        self.skipTest('Identity service does not support domain')
    role_name = self.role_prefix + '_grant_user_domain'
    role = self.operator_cloud.create_role(role_name)
    user_name = self.user_prefix + '_user_domain'
    user_email = 'nobody@nowhere.com'
    user = self._create_user(name=user_name, email=user_email, default_project='demo')
    self.assertTrue(self.operator_cloud.grant_role(role_name, user=user['id'], domain='default'))
    assignments = self.operator_cloud.list_role_assignments({'role': role['id'], 'user': user['id'], 'domain': self.operator_cloud.get_domain('default')['id']})
    self.assertIsInstance(assignments, list)
    self.assertEqual(1, len(assignments))
    self.assertTrue(self.operator_cloud.revoke_role(role_name, user=user['id'], domain='default'))
    assignments = self.operator_cloud.list_role_assignments({'role': role['id'], 'user': user['id'], 'domain': self.operator_cloud.get_domain('default')['id']})
    self.assertIsInstance(assignments, list)
    self.assertEqual(0, len(assignments))
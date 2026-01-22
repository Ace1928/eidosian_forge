from openstack import exceptions
from openstack.tests.functional import base
def test_update_user_password(self):
    user_name = self.user_prefix + '_password'
    user_email = 'nobody@nowhere.com'
    user = self._create_user(name=user_name, email=user_email, password='old_secret')
    self.assertIsNotNone(user)
    self.assertTrue(user['enabled'])
    new_user = self.operator_cloud.update_user(user['id'], password='new_secret')
    self.assertIsNotNone(new_user)
    self.assertEqual(user['id'], new_user['id'])
    self.assertEqual(user_name, new_user['name'])
    self.assertEqual(user_email, new_user['email'])
    self.assertTrue(new_user['enabled'])
    self.assertTrue(self.operator_cloud.grant_role('member', user=user['id'], project='demo', wait=True))
    self.addCleanup(self.operator_cloud.revoke_role, 'member', user=user['id'], project='demo', wait=True)
    new_cloud = self.operator_cloud.connect_as(user_id=user['id'], password='new_secret', project_name='demo')
    self.assertIsNotNone(new_cloud)
    location = new_cloud.current_location
    self.assertEqual(location['project']['name'], 'demo')
    self.assertIsNotNone(new_cloud.service_catalog)
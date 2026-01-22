from openstack import exceptions
from openstack.tests.functional import base
def test_create_qos_policy_basic(self):
    policy = self.operator_cloud.create_qos_policy(name=self.policy_name)
    self.assertIn('id', policy)
    self.assertEqual(self.policy_name, policy['name'])
    self.assertFalse(policy['is_shared'])
    self.assertFalse(policy['is_default'])
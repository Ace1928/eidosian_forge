import time
from testtools import content
from openstack.tests.functional import base
def test_update_cluster_policy(self):
    policy_name = 'example_policy'
    spec = {'properties': {'adjustment': {'min_step': 1, 'number': 1, 'type': 'CHANGE_IN_CAPACITY'}, 'event': 'CLUSTER_SCALE_IN'}, 'type': 'senlin.policy.scaling', 'version': '1.0'}
    self.addDetail('policy', content.text_content(policy_name))
    policy = self.user_cloud.create_cluster_policy(name=policy_name, spec=spec)
    self.addCleanup(self.cleanup_policy, policy['id'])
    policy_update = self.user_cloud.update_cluster_policy(policy['id'], new_name='new_policy_name')
    self.assertEqual(policy_update['policy']['id'], policy['id'])
    self.assertEqual(policy_update['policy']['spec'], policy['spec'])
    self.assertEqual(policy_update['policy']['name'], 'new_policy_name')
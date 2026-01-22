import time
from testtools import content
from openstack.tests.functional import base
def test_delete_cluster_policy(self):
    policy_name = 'example_policy'
    spec = {'properties': {'adjustment': {'min_step': 1, 'number': 1, 'type': 'CHANGE_IN_CAPACITY'}, 'event': 'CLUSTER_SCALE_IN'}, 'type': 'senlin.policy.scaling', 'version': '1.0'}
    self.addDetail('policy', content.text_content(policy_name))
    policy = self.user_cloud.create_cluster_policy(name=policy_name, spec=spec)
    self.addCleanup(self.cleanup_policy, policy['id'])
    policy_delete = self.user_cloud.delete_cluster_policy(policy['id'])
    self.assertTrue(policy_delete)
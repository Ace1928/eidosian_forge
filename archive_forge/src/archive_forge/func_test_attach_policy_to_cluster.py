import time
from testtools import content
from openstack.tests.functional import base
def test_attach_policy_to_cluster(self):
    profile_name = 'test_profile'
    spec = {'properties': {'flavor': self.flavor.name, 'image': self.image.name, 'networks': [{'network': 'private'}], 'security_groups': ['default']}, 'type': 'os.nova.server', 'version': 1.0}
    self.addDetail('profile', content.text_content(profile_name))
    profile = self.user_cloud.create_cluster_profile(name=profile_name, spec=spec)
    self.addCleanup(self.cleanup_profile, profile['id'])
    cluster_name = 'example_cluster'
    desired_capacity = 0
    self.addDetail('cluster', content.text_content(cluster_name))
    cluster = self.user_cloud.create_cluster(name=cluster_name, profile=profile, desired_capacity=desired_capacity)
    self.addCleanup(self.cleanup_cluster, cluster['cluster']['id'])
    policy_name = 'example_policy'
    spec = {'properties': {'adjustment': {'min_step': 1, 'number': 1, 'type': 'CHANGE_IN_CAPACITY'}, 'event': 'CLUSTER_SCALE_IN'}, 'type': 'senlin.policy.scaling', 'version': '1.0'}
    self.addDetail('policy', content.text_content(policy_name))
    policy = self.user_cloud.create_cluster_policy(name=policy_name, spec=spec)
    self.addCleanup(self.cleanup_policy, policy['id'], cluster['cluster']['id'])
    attach_cluster = self.user_cloud.get_cluster_by_id(cluster['cluster']['id'])
    attach_policy = self.user_cloud.get_cluster_policy_by_id(policy['id'])
    policy_attach = self.user_cloud.attach_policy_to_cluster(attach_cluster, attach_policy, is_enabled=True)
    self.assertTrue(policy_attach)
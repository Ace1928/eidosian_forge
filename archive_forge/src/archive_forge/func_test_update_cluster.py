import time
from testtools import content
from openstack.tests.functional import base
def test_update_cluster(self):
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
    self.user_cloud.update_cluster(cluster['cluster']['id'], new_name='new_cluster_name')
    wait = wait_for_status(self.user_cloud.get_cluster_by_id, {'name_or_id': cluster['cluster']['id']}, 'status', 'ACTIVE')
    self.assertTrue(wait)
    cluster_update = self.user_cloud.get_cluster_by_id(cluster['cluster']['id'])
    self.assertEqual(cluster_update['id'], cluster['cluster']['id'])
    self.assertEqual(cluster_update['name'], 'new_cluster_name')
    self.assertEqual(cluster_update['profile_id'], cluster['cluster']['profile_id'])
    self.assertEqual(cluster_update['desired_capacity'], cluster['cluster']['desired_capacity'])
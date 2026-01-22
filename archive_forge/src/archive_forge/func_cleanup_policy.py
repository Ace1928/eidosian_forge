import time
from testtools import content
from openstack.tests.functional import base
def cleanup_policy(self, name, cluster_name=None):
    if cluster_name is not None:
        cluster = self.user_cloud.get_cluster_by_id(cluster_name)
        policy = self.user_cloud.get_cluster_policy_by_id(name)
        policy_status = self.user_cloud.get_cluster_by_id(cluster['id'])['policies']
        if policy_status != []:
            self.user_cloud.detach_policy_from_cluster(cluster, policy)
    self.user_cloud.delete_cluster_policy(name)
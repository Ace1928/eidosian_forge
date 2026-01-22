import time
from testtools import content
from openstack.tests.functional import base
def cleanup_profile(self, name):
    time.sleep(5)
    for cluster in self.user_cloud.list_clusters():
        if name == cluster['profile_id']:
            self.user_cloud.delete_cluster(cluster['id'])
    self.user_cloud.delete_cluster_profile(name)
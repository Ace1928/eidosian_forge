import time
from testtools import content
from openstack.tests.functional import base
def cleanup_receiver(self, name):
    self.user_cloud.delete_cluster_receiver(name)
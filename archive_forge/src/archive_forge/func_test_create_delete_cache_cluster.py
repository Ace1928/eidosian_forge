import time
from tests.unit import unittest
from boto.elasticache import layer1
from boto.exception import BotoServerError
def test_create_delete_cache_cluster(self):
    cluster_id = 'cluster-id2'
    self.elasticache.create_cache_cluster(cluster_id, 1, 'cache.t1.micro', 'memcached')
    self.wait_until_cluster_available(cluster_id)
    self.elasticache.delete_cache_cluster(cluster_id)
    timeout = time.time() + 600
    while time.time() < timeout:
        try:
            self.elasticache.describe_cache_clusters(cluster_id)
        except BotoServerError:
            break
        time.sleep(5)
    else:
        self.fail('Timeout waiting for cache cluster %sto be deleted.' % cluster_id)
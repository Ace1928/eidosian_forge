import time
from tests.unit import unittest
from boto.elasticache import layer1
from boto.exception import BotoServerError
def wait_until_cluster_available(self, cluster_id):
    timeout = time.time() + 600
    while time.time() < timeout:
        response = self.elasticache.describe_cache_clusters(cluster_id)
        status = response['DescribeCacheClustersResponse']['DescribeCacheClustersResult']['CacheClusters'][0]['CacheClusterStatus']
        if status == 'available':
            break
        time.sleep(5)
    else:
        self.fail('Timeout waiting for cache cluster %rto become available.' % cluster_id)
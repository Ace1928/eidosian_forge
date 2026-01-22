import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def delete_cache_cluster(self, cache_cluster_id):
    """
        The DeleteCacheCluster operation deletes a previously
        provisioned cache cluster. DeleteCacheCluster deletes all
        associated cache nodes, node endpoints and the cache cluster
        itself. When you receive a successful response from this
        operation, Amazon ElastiCache immediately begins deleting the
        cache cluster; you cannot cancel or revert this operation.

        :type cache_cluster_id: string
        :param cache_cluster_id: The cache cluster identifier for the cluster
            to be deleted. This parameter is not case sensitive.

        """
    params = {'CacheClusterId': cache_cluster_id}
    return self._make_request(action='DeleteCacheCluster', verb='POST', path='/', params=params)
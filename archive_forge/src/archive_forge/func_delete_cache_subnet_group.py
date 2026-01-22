import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def delete_cache_subnet_group(self, cache_subnet_group_name):
    """
        The DeleteCacheSubnetGroup operation deletes a cache subnet
        group.
        You cannot delete a cache subnet group if it is associated
        with any cache clusters.

        :type cache_subnet_group_name: string
        :param cache_subnet_group_name: The name of the cache subnet group to
            delete.
        Constraints: Must contain no more than 255 alphanumeric characters or
            hyphens.

        """
    params = {'CacheSubnetGroupName': cache_subnet_group_name}
    return self._make_request(action='DeleteCacheSubnetGroup', verb='POST', path='/', params=params)
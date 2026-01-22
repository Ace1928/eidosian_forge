import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def delete_cache_parameter_group(self, cache_parameter_group_name):
    """
        The DeleteCacheParameterGroup operation deletes the specified
        cache parameter group. You cannot delete a cache parameter
        group if it is associated with any cache clusters.

        :type cache_parameter_group_name: string
        :param cache_parameter_group_name:
        The name of the cache parameter group to delete.

        The specified cache security group must not be associated with any
            cache clusters.

        """
    params = {'CacheParameterGroupName': cache_parameter_group_name}
    return self._make_request(action='DeleteCacheParameterGroup', verb='POST', path='/', params=params)
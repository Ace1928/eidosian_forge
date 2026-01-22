import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def delete_cache_security_group(self, cache_security_group_name):
    """
        The DeleteCacheSecurityGroup operation deletes a cache
        security group.
        You cannot delete a cache security group if it is associated
        with any cache clusters.

        :type cache_security_group_name: string
        :param cache_security_group_name:
        The name of the cache security group to delete.

        You cannot delete the default security group.

        """
    params = {'CacheSecurityGroupName': cache_security_group_name}
    return self._make_request(action='DeleteCacheSecurityGroup', verb='POST', path='/', params=params)
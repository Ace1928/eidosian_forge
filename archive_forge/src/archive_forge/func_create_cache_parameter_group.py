import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def create_cache_parameter_group(self, cache_parameter_group_name, cache_parameter_group_family, description):
    """
        The CreateCacheParameterGroup operation creates a new cache
        parameter group. A cache parameter group is a collection of
        parameters that you apply to all of the nodes in a cache
        cluster.

        :type cache_parameter_group_name: string
        :param cache_parameter_group_name: A user-specified name for the cache
            parameter group.

        :type cache_parameter_group_family: string
        :param cache_parameter_group_family: The name of the cache parameter
            group family the cache parameter group can be used with.
        Valid values are: `memcached1.4` | `redis2.6`

        :type description: string
        :param description: A user-specified description for the cache
            parameter group.

        """
    params = {'CacheParameterGroupName': cache_parameter_group_name, 'CacheParameterGroupFamily': cache_parameter_group_family, 'Description': description}
    return self._make_request(action='CreateCacheParameterGroup', verb='POST', path='/', params=params)
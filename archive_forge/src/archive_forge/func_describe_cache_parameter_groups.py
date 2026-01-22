import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def describe_cache_parameter_groups(self, cache_parameter_group_name=None, max_records=None, marker=None):
    """
        The DescribeCacheParameterGroups operation returns a list of
        cache parameter group descriptions. If a cache parameter group
        name is specified, the list will contain only the descriptions
        for that group.

        :type cache_parameter_group_name: string
        :param cache_parameter_group_name: The name of a specific cache
            parameter group to return details for.

        :type max_records: integer
        :param max_records: The maximum number of records to include in the
            response. If more records exist than the specified `MaxRecords`
            value, a marker is included in the response so that the remaining
            results can be retrieved.
        Default: 100

        Constraints: minimum 20; maximum 100.

        :type marker: string
        :param marker: An optional marker returned from a prior request. Use
            this marker for pagination of results from this operation. If this
            parameter is specified, the response includes only records beyond
            the marker, up to the value specified by MaxRecords .

        """
    params = {}
    if cache_parameter_group_name is not None:
        params['CacheParameterGroupName'] = cache_parameter_group_name
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeCacheParameterGroups', verb='POST', path='/', params=params)
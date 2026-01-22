import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def describe_cache_subnet_groups(self, cache_subnet_group_name=None, max_records=None, marker=None):
    """
        The DescribeCacheSubnetGroups operation returns a list of
        cache subnet group descriptions. If a subnet group name is
        specified, the list will contain only the description of that
        group.

        :type cache_subnet_group_name: string
        :param cache_subnet_group_name: The name of the cache subnet group to
            return details for.

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
    if cache_subnet_group_name is not None:
        params['CacheSubnetGroupName'] = cache_subnet_group_name
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeCacheSubnetGroups', verb='POST', path='/', params=params)
import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def describe_engine_default_parameters(self, cache_parameter_group_family, max_records=None, marker=None):
    """
        The DescribeEngineDefaultParameters operation returns the
        default engine and system parameter information for the
        specified cache engine.

        :type cache_parameter_group_family: string
        :param cache_parameter_group_family: The name of the cache parameter
            group family. Valid values are: `memcached1.4` | `redis2.6`

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
    params = {'CacheParameterGroupFamily': cache_parameter_group_family}
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeEngineDefaultParameters', verb='POST', path='/', params=params)
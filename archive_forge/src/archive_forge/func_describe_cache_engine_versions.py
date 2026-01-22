import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def describe_cache_engine_versions(self, engine=None, engine_version=None, cache_parameter_group_family=None, max_records=None, marker=None, default_only=None):
    """
        The DescribeCacheEngineVersions operation returns a list of
        the available cache engines and their versions.

        :type engine: string
        :param engine: The cache engine to return. Valid values: `memcached` |
            `redis`

        :type engine_version: string
        :param engine_version: The cache engine version to return.
        Example: `1.4.14`

        :type cache_parameter_group_family: string
        :param cache_parameter_group_family:
        The name of a specific cache parameter group family to return details
            for.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

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

        :type default_only: boolean
        :param default_only: If true , specifies that only the default version
            of the specified engine or engine and major version combination is
            to be returned.

        """
    params = {}
    if engine is not None:
        params['Engine'] = engine
    if engine_version is not None:
        params['EngineVersion'] = engine_version
    if cache_parameter_group_family is not None:
        params['CacheParameterGroupFamily'] = cache_parameter_group_family
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    if default_only is not None:
        params['DefaultOnly'] = str(default_only).lower()
    return self._make_request(action='DescribeCacheEngineVersions', verb='POST', path='/', params=params)
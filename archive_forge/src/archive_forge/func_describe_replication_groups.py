import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def describe_replication_groups(self, replication_group_id=None, max_records=None, marker=None):
    """
        The DescribeReplicationGroups operation returns information
        about a particular replication group. If no identifier is
        specified, DescribeReplicationGroups returns information about
        all replication groups.

        :type replication_group_id: string
        :param replication_group_id: The identifier for the replication group
            to be described. This parameter is not case sensitive.
        If you do not specify this parameter, information about all replication
            groups is returned.

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
    if replication_group_id is not None:
        params['ReplicationGroupId'] = replication_group_id
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeReplicationGroups', verb='POST', path='/', params=params)
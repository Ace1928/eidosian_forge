import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def describe_db_snapshots(self, db_instance_identifier=None, db_snapshot_identifier=None, snapshot_type=None, filters=None, max_records=None, marker=None):
    """
        Returns information about DB snapshots. This API supports
        pagination.

        :type db_instance_identifier: string
        :param db_instance_identifier:
        A DB instance identifier to retrieve the list of DB snapshots for.
            Cannot be used in conjunction with `DBSnapshotIdentifier`. This
            parameter is not case sensitive.

        Constraints:


        + Must contain from 1 to 63 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type db_snapshot_identifier: string
        :param db_snapshot_identifier:
        A specific DB snapshot identifier to describe. Cannot be used in
            conjunction with `DBInstanceIdentifier`. This value is stored as a
            lowercase string.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens
        + If this is the identifier of an automated snapshot, the
              `SnapshotType` parameter must also be specified.

        :type snapshot_type: string
        :param snapshot_type: The type of snapshots that will be returned.
            Values can be "automated" or "manual." If not specified, the
            returned results will include all snapshots types.

        :type filters: list
        :param filters:

        :type max_records: integer
        :param max_records: The maximum number of records to include in the
            response. If more records exist than the specified `MaxRecords`
            value, a pagination token called a marker is included in the
            response so that the remaining results may be retrieved.
        Default: 100

        Constraints: minimum 20, maximum 100

        :type marker: string
        :param marker: An optional pagination token provided by a previous
            `DescribeDBSnapshots` request. If this parameter is specified, the
            response includes only records beyond the marker, up to the value
            specified by `MaxRecords`.

        """
    params = {}
    if db_instance_identifier is not None:
        params['DBInstanceIdentifier'] = db_instance_identifier
    if db_snapshot_identifier is not None:
        params['DBSnapshotIdentifier'] = db_snapshot_identifier
    if snapshot_type is not None:
        params['SnapshotType'] = snapshot_type
    if filters is not None:
        self.build_complex_list_params(params, filters, 'Filters.member', ('FilterName', 'FilterValue'))
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeDBSnapshots', verb='POST', path='/', params=params)
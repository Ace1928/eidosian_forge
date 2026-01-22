import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def describe_db_instances(self, db_instance_identifier=None, filters=None, max_records=None, marker=None):
    """
        Returns information about provisioned RDS instances. This API
        supports pagination.

        :type db_instance_identifier: string
        :param db_instance_identifier:
        The user-supplied instance identifier. If this parameter is specified,
            information from only the specific DB instance is returned. This
            parameter isn't case sensitive.

        Constraints:


        + Must contain from 1 to 63 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

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
            DescribeDBInstances request. If this parameter is specified, the
            response includes only records beyond the marker, up to the value
            specified by `MaxRecords` .

        """
    params = {}
    if db_instance_identifier is not None:
        params['DBInstanceIdentifier'] = db_instance_identifier
    if filters is not None:
        self.build_complex_list_params(params, filters, 'Filters.member', ('FilterName', 'FilterValue'))
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeDBInstances', verb='POST', path='/', params=params)
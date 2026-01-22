import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def describe_db_security_groups(self, db_security_group_name=None, filters=None, max_records=None, marker=None):
    """
        Returns a list of `DBSecurityGroup` descriptions. If a
        `DBSecurityGroupName` is specified, the list will contain only
        the descriptions of the specified DB security group.

        :type db_security_group_name: string
        :param db_security_group_name: The name of the DB security group to
            return details for.

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
            DescribeDBSecurityGroups request. If this parameter is specified,
            the response includes only records beyond the marker, up to the
            value specified by `MaxRecords`.

        """
    params = {}
    if db_security_group_name is not None:
        params['DBSecurityGroupName'] = db_security_group_name
    if filters is not None:
        self.build_complex_list_params(params, filters, 'Filters.member', ('FilterName', 'FilterValue'))
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeDBSecurityGroups', verb='POST', path='/', params=params)
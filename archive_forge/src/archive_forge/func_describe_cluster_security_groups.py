import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def describe_cluster_security_groups(self, cluster_security_group_name=None, max_records=None, marker=None):
    """
        Returns information about Amazon Redshift security groups. If
        the name of a security group is specified, the response will
        contain only information about only that security group.

        For information about managing security groups, go to `Amazon
        Redshift Cluster Security Groups`_ in the Amazon Redshift
        Management Guide .

        :type cluster_security_group_name: string
        :param cluster_security_group_name: The name of a cluster security
            group for which you are requesting details. You can specify either
            the **Marker** parameter or a **ClusterSecurityGroupName**
            parameter, but not both.
        Example: `securitygroup1`

        :type max_records: integer
        :param max_records: The maximum number of response records to return in
            each call. If the number of remaining response records exceeds the
            specified `MaxRecords` value, a value is returned in a `marker`
            field of the response. You can retrieve the next set of records by
            retrying the command with the returned marker value.
        Default: `100`

        Constraints: minimum 20, maximum 100.

        :type marker: string
        :param marker: An optional parameter that specifies the starting point
            to return a set of response records. When the results of a
            DescribeClusterSecurityGroups request exceed the value specified in
            `MaxRecords`, AWS returns a value in the `Marker` field of the
            response. You can retrieve the next set of response records by
            providing the returned marker value in the `Marker` parameter and
            retrying the request.
        Constraints: You can specify either the **ClusterSecurityGroupName**
            parameter or the **Marker** parameter, but not both.

        """
    params = {}
    if cluster_security_group_name is not None:
        params['ClusterSecurityGroupName'] = cluster_security_group_name
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeClusterSecurityGroups', verb='POST', path='/', params=params)
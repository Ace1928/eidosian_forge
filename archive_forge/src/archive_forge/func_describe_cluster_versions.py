import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def describe_cluster_versions(self, cluster_version=None, cluster_parameter_group_family=None, max_records=None, marker=None):
    """
        Returns descriptions of the available Amazon Redshift cluster
        versions. You can call this operation even before creating any
        clusters to learn more about the Amazon Redshift versions. For
        more information about managing clusters, go to `Amazon
        Redshift Clusters`_ in the Amazon Redshift Management Guide

        :type cluster_version: string
        :param cluster_version: The specific cluster version to return.
        Example: `1.0`

        :type cluster_parameter_group_family: string
        :param cluster_parameter_group_family:
        The name of a specific cluster parameter group family to return details
            for.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

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
            DescribeClusterVersions request exceed the value specified in
            `MaxRecords`, AWS returns a value in the `Marker` field of the
            response. You can retrieve the next set of response records by
            providing the returned marker value in the `Marker` parameter and
            retrying the request.

        """
    params = {}
    if cluster_version is not None:
        params['ClusterVersion'] = cluster_version
    if cluster_parameter_group_family is not None:
        params['ClusterParameterGroupFamily'] = cluster_parameter_group_family
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeClusterVersions', verb='POST', path='/', params=params)
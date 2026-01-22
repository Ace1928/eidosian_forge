import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def describe_clusters(self, cluster_identifier=None, max_records=None, marker=None):
    """
        Returns properties of provisioned clusters including general
        cluster properties, cluster database properties, maintenance
        and backup properties, and security and access properties.
        This operation supports pagination. For more information about
        managing clusters, go to `Amazon Redshift Clusters`_ in the
        Amazon Redshift Management Guide .

        :type cluster_identifier: string
        :param cluster_identifier: The unique identifier of a cluster whose
            properties you are requesting. This parameter is case sensitive.
        The default is that all clusters defined for an account are returned.

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
            DescribeClusters request exceed the value specified in
            `MaxRecords`, AWS returns a value in the `Marker` field of the
            response. You can retrieve the next set of response records by
            providing the returned marker value in the `Marker` parameter and
            retrying the request.
        Constraints: You can specify either the **ClusterIdentifier** parameter
            or the **Marker** parameter, but not both.

        """
    params = {}
    if cluster_identifier is not None:
        params['ClusterIdentifier'] = cluster_identifier
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeClusters', verb='POST', path='/', params=params)
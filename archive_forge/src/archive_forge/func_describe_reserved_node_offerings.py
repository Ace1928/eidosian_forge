import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def describe_reserved_node_offerings(self, reserved_node_offering_id=None, max_records=None, marker=None):
    """
        Returns a list of the available reserved node offerings by
        Amazon Redshift with their descriptions including the node
        type, the fixed and recurring costs of reserving the node and
        duration the node will be reserved for you. These descriptions
        help you determine which reserve node offering you want to
        purchase. You then use the unique offering ID in you call to
        PurchaseReservedNodeOffering to reserve one or more nodes for
        your Amazon Redshift cluster.

        For more information about managing parameter groups, go to
        `Purchasing Reserved Nodes`_ in the Amazon Redshift Management
        Guide .

        :type reserved_node_offering_id: string
        :param reserved_node_offering_id: The unique identifier for the
            offering.

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
            DescribeReservedNodeOfferings request exceed the value specified in
            `MaxRecords`, AWS returns a value in the `Marker` field of the
            response. You can retrieve the next set of response records by
            providing the returned marker value in the `Marker` parameter and
            retrying the request.

        """
    params = {}
    if reserved_node_offering_id is not None:
        params['ReservedNodeOfferingId'] = reserved_node_offering_id
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeReservedNodeOfferings', verb='POST', path='/', params=params)
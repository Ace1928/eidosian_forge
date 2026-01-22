import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def describe_event_subscriptions(self, subscription_name=None, max_records=None, marker=None):
    """
        Lists descriptions of all the Amazon Redshift event
        notifications subscription for a customer account. If you
        specify a subscription name, lists the description for that
        subscription.

        :type subscription_name: string
        :param subscription_name: The name of the Amazon Redshift event
            notification subscription to be described.

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
            DescribeEventSubscriptions request exceed the value specified in
            `MaxRecords`, AWS returns a value in the `Marker` field of the
            response. You can retrieve the next set of response records by
            providing the returned marker value in the `Marker` parameter and
            retrying the request.

        """
    params = {}
    if subscription_name is not None:
        params['SubscriptionName'] = subscription_name
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeEventSubscriptions', verb='POST', path='/', params=params)
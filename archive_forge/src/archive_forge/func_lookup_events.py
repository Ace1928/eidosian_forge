import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudtrail import exceptions
from boto.compat import json
def lookup_events(self, lookup_attributes=None, start_time=None, end_time=None, max_results=None, next_token=None):
    """
        Looks up API activity events captured by CloudTrail that
        create, update, or delete resources in your account. Events
        for a region can be looked up for the times in which you had
        CloudTrail turned on in that region during the last seven
        days. Lookup supports five different attributes: time range
        (defined by a start time and end time), user name, event name,
        resource type, and resource name. All attributes are optional.
        The maximum number of attributes that can be specified in any
        one lookup request are time range and one other attribute. The
        default number of results returned is 10, with a maximum of 50
        possible. The response includes a token that you can use to
        get the next page of results.
        The rate of lookup requests is limited to one per second per
        account. If this limit is exceeded, a throttling error occurs.
        Events that occurred during the selected time range will not
        be available for lookup if CloudTrail logging was not enabled
        when the events occurred.

        :type lookup_attributes: list
        :param lookup_attributes: Contains a list of lookup attributes.
            Currently the list can contain only one item.

        :type start_time: timestamp
        :param start_time: Specifies that only events that occur after or at
            the specified time are returned. If the specified start time is
            after the specified end time, an error is returned.

        :type end_time: timestamp
        :param end_time: Specifies that only events that occur before or at the
            specified time are returned. If the specified end time is before
            the specified start time, an error is returned.

        :type max_results: integer
        :param max_results: The number of events to return. Possible values are
            1 through 50. The default is 10.

        :type next_token: string
        :param next_token: The token to use to get the next page of results
            after a previous API call. This token must be passed in with the
            same parameters that were specified in the the original call. For
            example, if the original call specified an AttributeKey of
            'Username' with a value of 'root', the call with NextToken should
            include those same parameters.

        """
    params = {}
    if lookup_attributes is not None:
        params['LookupAttributes'] = lookup_attributes
    if start_time is not None:
        params['StartTime'] = start_time
    if end_time is not None:
        params['EndTime'] = end_time
    if max_results is not None:
        params['MaxResults'] = max_results
    if next_token is not None:
        params['NextToken'] = next_token
    return self.make_request(action='LookupEvents', body=json.dumps(params))
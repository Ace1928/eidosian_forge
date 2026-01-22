import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.support import exceptions
def describe_communications(self, case_id, before_time=None, after_time=None, next_token=None, max_results=None):
    """
        Returns communications (and attachments) for one or more
        support cases. You can use the `AfterTime` and `BeforeTime`
        parameters to filter by date. You can use the `CaseId`
        parameter to restrict the results to a particular case.

        Case data is available for 12 months after creation. If a case
        was created more than 12 months ago, a request for data might
        cause an error.

        You can use the `MaxResults` and `NextToken` parameters to
        control the pagination of the result set. Set `MaxResults` to
        the number of cases you want displayed on each page, and use
        `NextToken` to specify the resumption of pagination.

        :type case_id: string
        :param case_id: The AWS Support case ID requested or returned in the
            call. The case ID is an alphanumeric string formatted as shown in
            this example: case- 12345678910-2013-c4c1d2bf33c5cf47

        :type before_time: string
        :param before_time: The end date for a filtered date search on support
            case communications. Case communications are available for 12
            months after creation.

        :type after_time: string
        :param after_time: The start date for a filtered date search on support
            case communications. Case communications are available for 12
            months after creation.

        :type next_token: string
        :param next_token: A resumption point for pagination.

        :type max_results: integer
        :param max_results: The maximum number of results to return before
            paginating.

        """
    params = {'caseId': case_id}
    if before_time is not None:
        params['beforeTime'] = before_time
    if after_time is not None:
        params['afterTime'] = after_time
    if next_token is not None:
        params['nextToken'] = next_token
    if max_results is not None:
        params['maxResults'] = max_results
    return self.make_request(action='DescribeCommunications', body=json.dumps(params))
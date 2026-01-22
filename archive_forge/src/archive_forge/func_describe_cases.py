import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.support import exceptions
def describe_cases(self, case_id_list=None, display_id=None, after_time=None, before_time=None, include_resolved_cases=None, next_token=None, max_results=None, language=None, include_communications=None):
    """
        Returns a list of cases that you specify by passing one or
        more case IDs. In addition, you can filter the cases by date
        by setting values for the `AfterTime` and `BeforeTime` request
        parameters.

        Case data is available for 12 months after creation. If a case
        was created more than 12 months ago, a request for data might
        cause an error.

        The response returns the following in JSON format:


        #. One or more CaseDetails data types.
        #. One or more `NextToken` values, which specify where to
           paginate the returned records represented by the `CaseDetails`
           objects.

        :type case_id_list: list
        :param case_id_list: A list of ID numbers of the support cases you want
            returned. The maximum number of cases is 100.

        :type display_id: string
        :param display_id: The ID displayed for a case in the AWS Support
            Center user interface.

        :type after_time: string
        :param after_time: The start date for a filtered date search on support
            case communications. Case communications are available for 12
            months after creation.

        :type before_time: string
        :param before_time: The end date for a filtered date search on support
            case communications. Case communications are available for 12
            months after creation.

        :type include_resolved_cases: boolean
        :param include_resolved_cases: Specifies whether resolved support cases
            should be included in the DescribeCases results. The default is
            false .

        :type next_token: string
        :param next_token: A resumption point for pagination.

        :type max_results: integer
        :param max_results: The maximum number of results to return before
            paginating.

        :type language: string
        :param language: The ISO 639-1 code for the language in which AWS
            provides support. AWS Support currently supports English ("en") and
            Japanese ("ja"). Language parameters must be passed explicitly for
            operations that take them.

        :type include_communications: boolean
        :param include_communications: Specifies whether communications should
            be included in the DescribeCases results. The default is true .

        """
    params = {}
    if case_id_list is not None:
        params['caseIdList'] = case_id_list
    if display_id is not None:
        params['displayId'] = display_id
    if after_time is not None:
        params['afterTime'] = after_time
    if before_time is not None:
        params['beforeTime'] = before_time
    if include_resolved_cases is not None:
        params['includeResolvedCases'] = include_resolved_cases
    if next_token is not None:
        params['nextToken'] = next_token
    if max_results is not None:
        params['maxResults'] = max_results
    if language is not None:
        params['language'] = language
    if include_communications is not None:
        params['includeCommunications'] = include_communications
    return self.make_request(action='DescribeCases', body=json.dumps(params))
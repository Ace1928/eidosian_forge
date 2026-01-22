import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def list_task_definitions(self, family_prefix=None, next_token=None, max_results=None):
    """
        Returns a list of task definitions that are registered to your
        account. You can filter the results by family name with the
        `familyPrefix` parameter.

        :type family_prefix: string
        :param family_prefix: The name of the family that you want to filter
            the `ListTaskDefinitions` results with. Specifying a `familyPrefix`
            will limit the listed task definitions to definitions that belong
            to that family.

        :type next_token: string
        :param next_token: The `nextToken` value returned from a previous
            paginated `ListTaskDefinitions` request where `maxResults` was used
            and the results exceeded the value of that parameter. Pagination
            continues from the end of the previous results that returned the
            `nextToken` value. This value is `null` when there are no more
            results to return.

        :type max_results: integer
        :param max_results: The maximum number of task definition results
            returned by `ListTaskDefinitions` in paginated output. When this
            parameter is used, `ListTaskDefinitions` only returns `maxResults`
            results in a single page along with a `nextToken` response element.
            The remaining results of the initial request can be seen by sending
            another `ListTaskDefinitions` request with the returned `nextToken`
            value. This value can be between 1 and 100. If this parameter is
            not used, then `ListTaskDefinitions` returns up to 100 results and
            a `nextToken` value if applicable.

        """
    params = {}
    if family_prefix is not None:
        params['familyPrefix'] = family_prefix
    if next_token is not None:
        params['nextToken'] = next_token
    if max_results is not None:
        params['maxResults'] = max_results
    return self._make_request(action='ListTaskDefinitions', verb='POST', path='/', params=params)
import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def list_clusters(self, next_token=None, max_results=None):
    """
        Returns a list of existing clusters.

        :type next_token: string
        :param next_token: The `nextToken` value returned from a previous
            paginated `ListClusters` request where `maxResults` was used and
            the results exceeded the value of that parameter. Pagination
            continues from the end of the previous results that returned the
            `nextToken` value. This value is `null` when there are no more
            results to return.

        :type max_results: integer
        :param max_results: The maximum number of cluster results returned by
            `ListClusters` in paginated output. When this parameter is used,
            `ListClusters` only returns `maxResults` results in a single page
            along with a `nextToken` response element. The remaining results of
            the initial request can be seen by sending another `ListClusters`
            request with the returned `nextToken` value. This value can be
            between 1 and 100. If this parameter is not used, then
            `ListClusters` returns up to 100 results and a `nextToken` value if
            applicable.

        """
    params = {}
    if next_token is not None:
        params['nextToken'] = next_token
    if max_results is not None:
        params['maxResults'] = max_results
    return self._make_request(action='ListClusters', verb='POST', path='/', params=params)
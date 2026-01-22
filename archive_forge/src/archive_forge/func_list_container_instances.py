import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def list_container_instances(self, cluster=None, next_token=None, max_results=None):
    """
        Returns a list of container instances in a specified cluster.

        :type cluster: string
        :param cluster: The short name or full Amazon Resource Name (ARN) of
            the cluster that hosts the container instances you want to list. If
            you do not specify a cluster, the default cluster is assumed..

        :type next_token: string
        :param next_token: The `nextToken` value returned from a previous
            paginated `ListContainerInstances` request where `maxResults` was
            used and the results exceeded the value of that parameter.
            Pagination continues from the end of the previous results that
            returned the `nextToken` value. This value is `null` when there are
            no more results to return.

        :type max_results: integer
        :param max_results: The maximum number of container instance results
            returned by `ListContainerInstances` in paginated output. When this
            parameter is used, `ListContainerInstances` only returns
            `maxResults` results in a single page along with a `nextToken`
            response element. The remaining results of the initial request can
            be seen by sending another `ListContainerInstances` request with
            the returned `nextToken` value. This value can be between 1 and
            100. If this parameter is not used, then `ListContainerInstances`
            returns up to 100 results and a `nextToken` value if applicable.

        """
    params = {}
    if cluster is not None:
        params['cluster'] = cluster
    if next_token is not None:
        params['nextToken'] = next_token
    if max_results is not None:
        params['maxResults'] = max_results
    return self._make_request(action='ListContainerInstances', verb='POST', path='/', params=params)
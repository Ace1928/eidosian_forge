import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def get_all_users(self, path_prefix='/', marker=None, max_items=None):
    """
        List the users that have the specified path prefix.

        :type path_prefix: string
        :param path_prefix: If provided, only users whose paths match
            the provided prefix will be returned.

        :type marker: string
        :param marker: Use this only when paginating results and only
            in follow-up request after you've received a response
            where the results are truncated.  Set this to the value of
            the Marker element in the response you just received.

        :type max_items: int
        :param max_items: Use this only when paginating results to indicate
            the maximum number of groups you want in the response.
        """
    params = {'PathPrefix': path_prefix}
    if marker:
        params['Marker'] = marker
    if max_items:
        params['MaxItems'] = max_items
    return self.get_response('ListUsers', params, list_marker='Users')
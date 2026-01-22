import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def list_policies(self, marker=None, max_items=None, only_attached=None, path_prefix=None, scope=None):
    """
        List policies of account.

        :type marker: string
        :param marker: A marker used for pagination (received from previous
            accesses)

        :type max_items: int
        :param max_items: Send only max_items; allows paginations

        :type only_attached: bool
        :param only_attached: Send only policies attached to other resources

        :type path_prefix: string
        :param path_prefix: Send only items prefixed by this path

        :type scope: string
        :param scope: AWS|Local.  Choose between AWS policies or your own
        """
    params = {}
    if path_prefix is not None:
        params['PathPrefix'] = path_prefix
    if marker is not None:
        params['Marker'] = marker
    if max_items is not None:
        params['MaxItems'] = max_items
    if type(only_attached) == bool:
        params['OnlyAttached'] = str(only_attached).lower()
    if scope is not None:
        params['Scope'] = scope
    return self.get_response('ListPolicies', params, list_marker='Policies')
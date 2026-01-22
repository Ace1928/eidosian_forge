import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_targethttpsproxy_set_urlmap(self, targethttpsproxy, urlmap):
    """
        Changes the URL map for TargetHttpsProxy.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  targethttpsproxy:  Name of the TargetHttpsProxy resource
                                   whose URL map is to be set.
        :type   targethttpsproxy: ``str``

        :param  urlmap:  urlmap to set.
        :type   urlmap: :class:`GCEUrlMap`

        :return:  True
        :rtype: ``bool``
        """
    request = '/targetHttpsProxies/%s/setUrlMap' % targethttpsproxy.name
    request_data = {'urlMap': urlmap.extra['selfLink']}
    self.connection.async_request(request, method='POST', data=request_data)
    return True
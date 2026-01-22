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
def ex_get_targethttpproxy(self, name):
    """
        Return a Target HTTP Proxy object based on its name.

        :param  name: The name of the target HTTP proxy.
        :type   name: ``str``

        :return:  A Target HTTP Proxy object for the pool
        :rtype:   :class:`GCETargetHttpProxy`
        """
    request = '/global/targetHttpProxies/%s' % name
    response = self.connection.request(request, method='GET').object
    return self._to_targethttpproxy(response)
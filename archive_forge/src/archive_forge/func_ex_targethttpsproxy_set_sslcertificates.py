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
def ex_targethttpsproxy_set_sslcertificates(self, targethttpsproxy, sslcertificates):
    """
        Replaces SslCertificates for TargetHttpsProxy.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  targethttpsproxy:  Name of the TargetHttpsProxy resource to
                                   set an SslCertificates resource for.
        :type   targethttpsproxy: ``str``

        :param  sslcertificates:  sslcertificates to set.
        :type   sslcertificates: ``list`` of :class:`GCESslCertificates`

        :return:  True
        :rtype: ``bool``
        """
    request = '/targetHttpsProxies/%s/setSslCertificates' % targethttpsproxy.name
    request_data = {'sslCertificates': [x.extra['selfLink'] for x in sslcertificates]}
    self.connection.async_request(request, method='POST', data=request_data)
    return True
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
def ex_create_targethttpsproxy(self, name, urlmap, sslcertificates, description=None):
    """
        Creates a TargetHttpsProxy resource in the specified project
        using the data included in the request.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  name:  Name of the resource. Provided by the client when the
                       resource is created. The name must be 1-63 characters
                       long, and comply with RFC1035. Specifically, the name
                       must be 1-63 characters long and match the regular
                       expression [a-z]([-a-z0-9]*[a-z0-9])? which means the
                       first character must be a lowercase letter, and all
                       following characters must be a dash, lowercase letter,
                       or digit, except the last character, which cannot be a
                       dash.
        :type   name: ``str``

        :param  sslcertificates:  URLs to SslCertificate resources that
                                     are used to authenticate connections
                                     between users and the load balancer.
                                     Currently, exactly one SSL certificate
                                     must be specified.
        :type   sslcertificates: ``list`` of :class:`GCESslcertificates`

        :param  urlmap:  A fully-qualified or valid partial URL to the
                            UrlMap resource that defines the mapping from URL
                            to the BackendService.
        :type   urlmap: :class:`GCEUrlMap`

        :keyword  description:  An optional description of this resource.
                                Provide this property when you create the
                                resource.
        :type   description: ``str``

        :return:  `GCETargetHttpsProxy` object.
        :rtype: :class:`GCETargetHttpsProxy`
        """
    request = '/global/targetHttpsProxies'
    request_data = {}
    request_data['name'] = name
    request_data['description'] = description
    request_data['sslCertificates'] = [x.extra['selfLink'] for x in sslcertificates]
    request_data['urlMap'] = urlmap.extra['selfLink']
    self.connection.async_request(request, method='POST', data=request_data)
    return self.ex_get_targethttpsproxy(name)
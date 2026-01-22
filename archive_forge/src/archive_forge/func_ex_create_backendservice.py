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
def ex_create_backendservice(self, name, healthchecks, backends=[], protocol=None, description=None, timeout_sec=None, enable_cdn=False, port=None, port_name=None):
    """
        Create a global Backend Service.

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

        :param    healthchecks: A list of HTTP Health Checks to use for this
                                service.  There must be at least one.
        :type     healthchecks: ``list`` of (``str`` or
                                :class:`GCEHealthCheck`)

        :keyword  backends:  The list of backends that serve this
                             BackendService.
        :type   backends: ``list`` of :class `GCEBackend` or list of ``dict``

        :keyword  timeout_sec:  How many seconds to wait for the backend
                                before considering it a failed request.
                                Default is 30 seconds.
        :type   timeout_sec: ``integer``

        :keyword  enable_cdn:  If true, enable Cloud CDN for this
                                 BackendService.  When the load balancing
                                 scheme is INTERNAL, this field is not used.
        :type   enable_cdn: ``bool``

        :keyword  port:  Deprecated in favor of port_name. The TCP port to
                         connect on the backend. The default value is 80.
                         This cannot be used for internal load balancing.
        :type   port: ``integer``

        :keyword  port_name: Name of backend port. The same name should appear
                             in the instance groups referenced by this service.
        :type     port_name: ``str``

        :keyword  protocol: The protocol this Backend Service uses to
                            communicate with backends.
                            Possible values are HTTP, HTTPS, HTTP2, TCP
                            and SSL.
        :type     protocol: ``str``

        :return:  A Backend Service object.
        :rtype:   :class:`GCEBackendService`
        """
    backendservice_data = {'name': name, 'healthChecks': [], 'backends': [], 'enableCDN': enable_cdn}
    for hc in healthchecks:
        if not hasattr(hc, 'extra'):
            hc = self.ex_get_healthcheck(name=hc)
        backendservice_data['healthChecks'].append(hc.extra['selfLink'])
    for be in backends:
        if isinstance(be, GCEBackend):
            backendservice_data['backends'].append(be.to_backend_dict())
        else:
            backendservice_data['backends'].append(be)
    if port:
        backendservice_data['port'] = port
    if port_name:
        backendservice_data['portName'] = port_name
    if timeout_sec:
        backendservice_data['timeoutSec'] = timeout_sec
    if protocol:
        if protocol in self.BACKEND_SERVICE_PROTOCOLS:
            backendservice_data['protocol'] = protocol
        else:
            raise ValueError('Protocol must be one of %s' % ','.join(self.BACKEND_SERVICE_PROTOCOLS))
    if description:
        backendservice_data['description'] = description
    request = '/global/backendServices'
    self.connection.async_request(request, method='POST', data=backendservice_data)
    return self.ex_get_backendservice(name)
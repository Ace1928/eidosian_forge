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
def ex_create_urlmap(self, name, default_service):
    """
        Create a URL Map.

        :param  name: Name of the URL Map.
        :type   name: ``str``

        :keyword  default_service: Default backend service for the map.
        :type     default_service: ``str`` or :class:`GCEBackendService`

        :return:  URL Map object
        :rtype:   :class:`GCEUrlMap`
        """
    urlmap_data = {'name': name}
    if not hasattr(default_service, 'name'):
        default_service = self.ex_get_backendservice(default_service)
    urlmap_data['defaultService'] = default_service.extra['selfLink']
    request = '/global/urlMaps'
    self.connection.async_request(request, method='POST', data=urlmap_data)
    return self.ex_get_urlmap(name)
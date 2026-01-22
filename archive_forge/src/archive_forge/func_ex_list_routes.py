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
def ex_list_routes(self):
    """
        Return the list of routes.

        :return: A list of route objects.
        :rtype: ``list`` of :class:`GCERoute`
        """
    list_routes = []
    request = '/global/routes'
    response = self.connection.request(request, method='GET').object
    list_routes = [self._to_route(n) for n in response.get('items', [])]
    return list_routes
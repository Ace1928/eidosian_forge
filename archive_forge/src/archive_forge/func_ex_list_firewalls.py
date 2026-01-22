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
def ex_list_firewalls(self):
    """
        Return the list of firewalls.

        :return: A list of firewall objects.
        :rtype: ``list`` of :class:`GCEFirewall`
        """
    list_firewalls = []
    request = '/global/firewalls'
    response = self.connection.request(request, method='GET').object
    list_firewalls = [self._to_firewall(f) for f in response.get('items', [])]
    return list_firewalls
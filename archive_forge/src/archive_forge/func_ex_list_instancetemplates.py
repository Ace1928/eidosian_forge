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
def ex_list_instancetemplates(self):
    """
        Return the list of Instance Templates.

        :return:  A list of Instance Template Objects
        :rtype:   ``list`` of :class:`GCEInstanceTemplate`
        """
    request = '/global/instanceTemplates'
    response = self.connection.request(request, method='GET').object
    return [self._to_instancetemplate(u) for u in response.get('items', [])]
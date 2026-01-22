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
def ex_get_project(self):
    """
        Return a Project object with project-wide information.

        :return:  A GCEProject object
        :rtype:   :class:`GCEProject`
        """
    response = self.connection.request('', method='GET').object
    return self._to_project(response)
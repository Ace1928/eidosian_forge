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
def get_health(self, node=None):
    """
        Return a hash of target pool instances and their health.

        :param  node: Optional node to specify if only a specific node's
                      health status should be returned
        :type   node: ``str``, ``Node``, or ``None``

        :return: List of hashes of nodes and their respective health
        :rtype:  ``list`` of ``dict``
        """
    return self.driver.ex_targetpool_get_health(targetpool=self, node=node)
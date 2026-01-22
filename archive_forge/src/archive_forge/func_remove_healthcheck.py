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
def remove_healthcheck(self, healthcheck):
    """
        Remove a healthcheck from this target pool.

        :param  healthcheck: Healthcheck to remove
        :type   healthcheck: ``str`` or :class:`GCEHealthCheck`

        :return:  True if successful
        :rtype:   ``bool``
        """
    return self.driver.ex_targetpool_remove_healthcheck(targetpool=self, healthcheck=healthcheck)
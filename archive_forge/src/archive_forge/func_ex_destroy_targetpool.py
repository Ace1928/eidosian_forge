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
def ex_destroy_targetpool(self, targetpool):
    """
        Destroy a target pool.

        :param  targetpool: TargetPool object to destroy
        :type   targetpool: :class:`GCETargetPool`

        :return:  True if successful
        :rtype:   ``bool``
        """
    request = '/regions/{}/targetPools/{}'.format(targetpool.region.name, targetpool.name)
    self.connection.async_request(request, method='DELETE')
    return True
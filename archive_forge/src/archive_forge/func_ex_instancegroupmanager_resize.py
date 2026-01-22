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
def ex_instancegroupmanager_resize(self, manager, size):
    """
        Set the Instance Template for this Instance Group.

        :param  manager: Instance Group Manager to operate on.
        :type   manager: :class:`GCEInstanceGroupManager`

        :param  size: New size of Managed Instance Group.
        :type   size: ``int``

        :return:  True if successful
        :rtype:   ``bool``
        """
    req_params = {'size': size}
    request = '/zones/{}/instanceGroupManagers/{}/resize'.format(manager.zone.name, manager.name)
    self.connection.async_request(request, method='POST', params=req_params)
    return True
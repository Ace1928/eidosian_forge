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
def ex_instancegroupmanager_set_instancetemplate(self, manager, instancetemplate):
    """
        Set the Instance Template for this Instance Group.  Existing VMs are
        not recreated by setting a new InstanceTemplate.

        :param  manager: Instance Group Manager to operate on.
        :type   manager: :class:`GCEInstanceGroupManager`

        :param  instancetemplate: Instance Template to set.
        :type   instancetemplate: :class:`GCEInstanceTemplate`

        :return:  True if successful
        :rtype:   ``bool``
        """
    req_data = {'instanceTemplate': instancetemplate.extra['selfLink']}
    request = '/zones/%s/instanceGroupManagers/%s/setInstanceTemplate' % (manager.zone.name, manager.name)
    self.connection.async_request(request, method='POST', data=req_data)
    return True
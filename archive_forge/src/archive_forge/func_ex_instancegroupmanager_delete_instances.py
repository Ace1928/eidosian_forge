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
def ex_instancegroupmanager_delete_instances(self, manager, node_list):
    """
        Remove instances from GCEInstanceGroupManager and destroy
        the instance

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  manager:  Required. The name of the managed instance group. The
                       name must be 1-63 characters long, and comply with
                       RFC1035.
        :type   manager: ``str`` or :class: `GCEInstanceGroupManager`

        :param  node_list:  list of Node objects to delete.
        :type   node_list: ``list`` of :class:`Node`

        :return:  True if successful
        :rtype: ``bool``
        """
    request = '/zones/{}/instanceGroupManagers/{}/deleteInstances'.format(manager.zone.name, manager.name)
    request_data = {'instances': [x.extra['selfLink'] for x in node_list]}
    self.connection.request(request, method='POST', data=request_data).object
    return True
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
def ex_instancegroup_list_instances(self, instancegroup):
    """
        Lists the instances in the specified instance group.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute
        * https://www.googleapis.com/auth/compute.readonly

        :param  instancegroup:  The Instance Group where from which you
                                want to generate a list of included
                                instances.
        :type   instancegroup: :class:`GCEInstanceGroup`

        :return:  List of :class:`GCENode` objects.
        :rtype: ``list`` of :class:`GCENode` objects.
        """
    request = '/zones/{}/instanceGroups/{}/listInstances'.format(instancegroup.zone.name, instancegroup.name)
    response = self.connection.request(request, method='POST').object
    list_data = []
    if 'items' in response:
        for v in response['items']:
            instance_info = self._get_components_from_path(v['instance'])
            list_data.append(self.ex_get_node(instance_info['name'], instance_info['zone']))
    return list_data
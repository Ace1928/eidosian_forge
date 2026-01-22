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
def ex_list_instancegroups(self, zone):
    """
        Retrieves the list of instance groups that are located in the specified
        project and zone.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute
        * https://www.googleapis.com/auth/compute.readonly

        :param  zone:  The name of the zone where the instance group is
                       located.
        :type   zone: ``str``

        :return: A list of instance group mgr  objects.
        :rtype: ``list`` of :class:`GCEInstanceGroupManagers`
        """
    list_data = []
    zone = self._set_zone(zone)
    if zone is None:
        request = '/aggregated/instanceGroups'
    else:
        request = '/zones/%s/instanceGroups' % zone.name
    response = self.connection.request(request, method='GET').object
    if 'items' in response:
        if zone is None:
            for v in response['items'].values():
                zone_data = [self._to_instancegroup(a) for a in v.get('instanceGroups', [])]
                list_data.extend(zone_data)
        else:
            list_data = [self._to_instancegroup(a) for a in response['items']]
    return list_data
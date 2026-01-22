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
def ex_list_autoscalers(self, zone=None):
    """
        Return the list of AutoScalers.

        :keyword  zone: The zone to return InstanceGroupManagers from.
                        For example: 'us-central1-a'.  If None, will return
                        InstanceGroupManagers from self.zone.  If 'all', will
                        return all InstanceGroupManagers.
        :type     zone: ``str`` or ``None``

        :return:  A list of AutoScaler Objects
        :rtype:   ``list`` of :class:`GCEAutoScaler`
        """
    list_autoscalers = []
    zone = self._set_zone(zone)
    if zone is None:
        request = '/aggregated/autoscalers'
    else:
        request = '/zones/%s/autoscalers' % zone.name
    response = self.connection.request(request, method='GET').object
    if 'items' in response:
        if zone is None:
            for v in response['items'].values():
                zone_as = [self._to_autoscaler(a) for a in v.get('autoscalers', [])]
                list_autoscalers.extend(zone_as)
        else:
            list_autoscalers = [self._to_autoscaler(a) for a in response['items']]
    return list_autoscalers
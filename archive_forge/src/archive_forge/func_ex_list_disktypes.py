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
def ex_list_disktypes(self, zone=None):
    """
        Return a list of DiskTypes for a zone or all.

        :keyword  zone: The zone to return DiskTypes from. For example:
                        'us-central1-a'.  If None, will return DiskTypes from
                        self.zone.  If 'all', will return all DiskTypes.
        :type     zone: ``str`` or ``None``

        :return: A list of static DiskType objects.
        :rtype: ``list`` of :class:`GCEDiskType`
        """
    list_disktypes = []
    zone = self._set_zone(zone)
    if zone is None:
        request = '/aggregated/diskTypes'
    else:
        request = '/zones/%s/diskTypes' % zone.name
    response = self.connection.request(request, method='GET').object
    if 'items' in response:
        if zone is None:
            for v in response['items'].values():
                zone_disktypes = [self._to_disktype(a) for a in v.get('diskTypes', [])]
                list_disktypes.extend(zone_disktypes)
        else:
            list_disktypes = [self._to_disktype(a) for a in response['items']]
    return list_disktypes
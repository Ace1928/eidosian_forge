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
def ex_get_targetinstance(self, name, zone=None):
    """
        Return a TargetInstance object based on a name and optional zone.

        :param  name: The name of the target instance
        :type   name: ``str``

        :keyword  zone: The zone to search for the target instance in (set to
                          'all' to search all zones).
        :type     zone: ``str`` or :class:`GCEZone` or ``None``

        :return:  A TargetInstance object for the instance
        :rtype:   :class:`GCETargetInstance`
        """
    zone = self._set_zone(zone) or self._find_zone_or_region(name, 'targetInstances', res_name='TargetInstance')
    request = '/zones/{}/targetInstances/{}'.format(zone.name, name)
    response = self.connection.request(request, method='GET').object
    return self._to_targetinstance(response)
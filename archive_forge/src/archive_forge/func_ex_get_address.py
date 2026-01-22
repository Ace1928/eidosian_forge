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
def ex_get_address(self, name, region=None):
    """
        Return an Address object based on an address name and optional region.

        :param  name: The name of the address
        :type   name: ``str``

        :keyword  region: The region to search for the address in (set to
                          'all' to search all regions)
        :type     region: ``str`` :class:`GCERegion` or ``None``

        :return:  An Address object for the address
        :rtype:   :class:`GCEAddress`
        """
    if region == 'global':
        request = '/global/addresses/%s' % name
    else:
        region = self._set_region(region) or self._find_zone_or_region(name, 'addresses', region=True, res_name='Address')
        request = '/regions/{}/addresses/{}'.format(region.name, name)
    response = self.connection.request(request, method='GET').object
    return self._to_address(response)
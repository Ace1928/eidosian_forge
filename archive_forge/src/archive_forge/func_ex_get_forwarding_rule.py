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
def ex_get_forwarding_rule(self, name, region=None, global_rule=False):
    """
        Return a Forwarding Rule object based on the forwarding rule name.

        :param  name: The name of the forwarding rule
        :type   name: ``str``

        :keyword  region: The region to search for the rule in (set to 'all'
                          to search all regions).
        :type     region: ``str`` or ``None``

        :keyword  global_rule: Set to True to get a global forwarding rule.
                                Region will be ignored if True.
        :type     global_rule: ``bool``

        :return:  A GCEForwardingRule object
        :rtype:   :class:`GCEForwardingRule`
        """
    if global_rule:
        request = '/global/forwardingRules/%s' % name
    else:
        region = self._set_region(region) or self._find_zone_or_region(name, 'forwardingRules', region=True, res_name='ForwardingRule')
        request = '/regions/{}/forwardingRules/{}'.format(region.name, name)
    response = self.connection.request(request, method='GET').object
    return self._to_forwarding_rule(response)
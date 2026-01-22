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
def ex_delete_access_config(self, node, name, nic):
    """
        Delete a network interface access configuration from a node.

        :keyword  node: The existing target Node (instance) for the request.
        :type     node: ``Node``

        :keyword  name: Name of the access config.
        :type     name: ``str``

        :keyword  nic: Name of the network interface.
        :type     nic: ``str``

        :return: True if successful
        :rtype:  ``bool``
        """
    if not isinstance(node, Node):
        raise ValueError('Must specify a valid libcloud node object.')
    node_name = node.name
    zone_name = node.extra['zone'].name
    params = {'accessConfig': name, 'networkInterface': nic}
    request = '/zones/{}/instances/{}/deleteAccessConfig'.format(zone_name, node_name)
    self.connection.async_request(request, method='POST', params=params)
    return True
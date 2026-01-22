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
def ex_add_access_config(self, node, name, nic, nat_ip=None, config_type=None):
    """
        Add a network interface access configuration to a node.

        :keyword  node: The existing target Node (instance) that will receive
                        the new access config.
        :type     node: ``Node``

        :keyword  name: Name of the new access config.
        :type     node: ``str``

        :keyword  nat_ip: The external existing static IP Address to use for
                          the access config. If not provided, an ephemeral
                          IP address will be allocated.
        :type     nat_ip: ``str`` or ``None``

        :keyword  config_type: The type of access config to create. Currently
                               the only supported type is 'ONE_TO_ONE_NAT'.
        :type     config_type: ``str`` or ``None``

        :return: True if successful
        :rtype:  ``bool``
        """
    if not isinstance(node, Node):
        raise ValueError('Must specify a valid libcloud node object.')
    node_name = node.name
    zone_name = node.extra['zone'].name
    config = {'name': name}
    if config_type is None:
        config_type = 'ONE_TO_ONE_NAT'
    config['type'] = config_type
    if nat_ip is not None:
        config['natIP'] = nat_ip
    params = {'networkInterface': nic}
    request = '/zones/{}/instances/{}/addAccessConfig'.format(zone_name, node_name)
    self.connection.async_request(request, method='POST', data=config, params=params)
    return True
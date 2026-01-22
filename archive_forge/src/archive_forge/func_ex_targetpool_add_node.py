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
def ex_targetpool_add_node(self, targetpool, node):
    """
        Add a node to a target pool.

        :param  targetpool: The targetpool to add node to
        :type   targetpool: ``str`` or :class:`GCETargetPool`

        :param  node: The node to add
        :type   node: ``str`` or :class:`Node`

        :return: True if successful
        :rtype:  ``bool``
        """
    if not hasattr(targetpool, 'name'):
        targetpool = self.ex_get_targetpool(targetpool)
    if hasattr(node, 'name'):
        node_uri = node.extra['selfLink']
    elif node.startswith('https://'):
        node_uri = node
    else:
        node = self.ex_get_node(node, 'all')
        node_uri = node.extra['selfLink']
    targetpool_data = {'instances': [{'instance': node_uri}]}
    request = '/regions/{}/targetPools/{}/addInstance'.format(targetpool.region.name, targetpool.name)
    self.connection.async_request(request, method='POST', data=targetpool_data)
    if all((node_uri != n and (not hasattr(n, 'extra') or n.extra['selfLink'] != node_uri) for n in targetpool.nodes)):
        targetpool.nodes.append(node)
    return True
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
def ex_targetpool_remove_node(self, targetpool, node):
    """
        Remove a node from a target pool.

        :param  targetpool: The targetpool to remove node from
        :type   targetpool: ``str`` or :class:`GCETargetPool`

        :param  node: The node to remove
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
    request = '/regions/{}/targetPools/{}/removeInstance'.format(targetpool.region.name, targetpool.name)
    self.connection.async_request(request, method='POST', data=targetpool_data)
    index = None
    for i, nd in enumerate(targetpool.nodes):
        if nd == node_uri or (hasattr(nd, 'extra') and nd.extra['selfLink'] == node_uri):
            index = i
            break
    if index is not None:
        targetpool.nodes.pop(index)
    return True
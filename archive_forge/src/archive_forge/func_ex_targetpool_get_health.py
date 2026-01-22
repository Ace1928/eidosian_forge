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
def ex_targetpool_get_health(self, targetpool, node=None):
    """
        Return a hash of target pool instances and their health.

        :param  targetpool: Targetpool containing healthchecked instances.
        :type   targetpool: :class:`GCETargetPool`

        :param  node: Optional node to specify if only a specific node's
                      health status should be returned
        :type   node: ``str``, ``Node``, or ``None``

        :return: List of hashes of instances and their respective health,
                 e.g. [{'node': ``Node``, 'health': 'UNHEALTHY'}, ...]
        :rtype:  ``list`` of ``dict``
        """
    health = []
    region_name = targetpool.region.name
    request = '/regions/{}/targetPools/{}/getHealth'.format(region_name, targetpool.name)
    if node is not None:
        if hasattr(node, 'name'):
            node_name = node.name
        else:
            node_name = node
    nodes = targetpool.nodes
    for node_object in nodes:
        if node:
            if node_name == node_object.name:
                body = {'instance': node_object.extra['selfLink']}
                resp = self.connection.request(request, method='POST', data=body).object
                status = resp['healthStatus'][0]['healthState']
                health.append({'node': node_object, 'health': status})
        else:
            body = {'instance': node_object.extra['selfLink']}
            resp = self.connection.request(request, method='POST', data=body).object
            status = resp['healthStatus'][0]['healthState']
            health.append({'node': node_object, 'health': status})
    return health
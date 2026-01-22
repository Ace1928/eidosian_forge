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
def ex_set_node_tags(self, node, tags):
    """
        Set the tags on a Node instance.

        Note that this updates the node object directly.

        :param  node: Node object
        :type   node: :class:`Node`

        :param  tags: List of tags to apply to the object
        :type   tags: ``list`` of ``str``

        :return:  True if successful
        :rtype:   ``bool``
        """
    request = '/zones/{}/instances/{}/setTags'.format(node.extra['zone'].name, node.name)
    tags_data = {}
    tags_data['items'] = tags
    tags_data['fingerprint'] = node.extra['tags_fingerprint']
    self.connection.async_request(request, method='POST', data=tags_data)
    new_node = self.ex_get_node(node.name, node.extra['zone'])
    node.extra['tags'] = new_node.extra['tags']
    node.extra['tags_fingerprint'] = new_node.extra['tags_fingerprint']
    return True
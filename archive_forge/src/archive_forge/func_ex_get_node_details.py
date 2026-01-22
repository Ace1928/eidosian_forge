import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
def ex_get_node_details(self, node_id):
    """
        Lists details of the specified server.

        :param       node_id: ID of the node which should be used
        :type        node_id: ``str``

        :rtype: :class:`Node`
        """
    data = self._paginated_request('/v2/droplets/{}'.format(node_id), 'droplet')
    return self._to_node(data)
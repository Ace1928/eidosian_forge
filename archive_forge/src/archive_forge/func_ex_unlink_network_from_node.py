import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_unlink_network_from_node(self, node, network):
    """
        Unlink network from node.

        :param node: Node you want to unlink from network.
        :type node: :class:`.Node`

        :param network: Network you want to unlink.
        :type network: :class:`.GridscaleNetwork

        :return: ``True`` if unlink was successful, otherwise ``False``
        :rtype: ``bool``
        """
    result = self._sync_request(endpoint='objects/servers/{}/networks/{}'.format(node.id, network.id), method='DELETE')
    return result.status == 204
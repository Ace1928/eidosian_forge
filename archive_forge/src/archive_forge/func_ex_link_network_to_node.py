import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_link_network_to_node(self, node, network):
    """
        Link a network to a node.

        :param node: Node object to link networks to.
        :type node: :class:`.Node`

        :param network: Network you want to link.
        :type network: :class:`.GridscaleNetwork`

        :return: ``True`` if linked successfully, otherwise ``False``
        :rtype: ``bool``
        """
    result = self._sync_request(data={'object_uuid': network.id}, endpoint='objects/servers/{}/networks/'.format(node.id), method='POST')
    return result.status == 204
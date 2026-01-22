from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_node_attach_interface(self, node, iface):
    """
        Specific method to attach an interface to a node

        :param      node: Node which should be used
        :type       node: :class:`Node`

        :param      iface: Network interface which should be used
        :type       iface: :class:`GandiNetworkInterface`

        :rtype: ``bool``
        """
    op = self.connection.request('hosting.vm.iface_attach', int(node.id), int(iface.id))
    if self._wait_operation(op.object['id']):
        return True
    return False